"""
Structure Matrix Builder
========================

Constructs structure matrices for individual neurons, layers,
and entire networks. This is the computational heart of SNA-STP.

A structure matrix M ∈ {0,1}^{2^m × 2^n} maps every possible
boolean input combination to the corresponding boolean output,
providing a complete logical characterization of the function.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from .stp import STPCore


class StructureMatrixBuilder:
    """
    Build structure matrices at neuron, layer, and network levels.

    The structure matrix captures the complete input-output boolean
    mapping of a neural network component after quantization.

    Supported activation functions: sigmoid, relu, tanh, step, linear.

    Examples:
        >>> builder = StructureMatrixBuilder(threshold=0.5)
        >>> # XOR: weights = [-1, -1], bias = 1.5 (approximate)
        >>> M = builder.build_neuron_matrix(
        ...     weights=np.array([1.0, 1.0]),
        ...     bias=-1.5,
        ...     activation='sigmoid'
        ... )
        >>> print(M.shape)  # (2, 4) for 2 inputs
    """

    ACTIVATIONS = {
        "sigmoid": lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))),
        "relu": lambda z: np.maximum(0, z),
        "tanh": lambda z: np.tanh(z),
        "step": lambda z: float(z >= 0),
        "linear": lambda z: z,
    }

    def __init__(self, threshold: float = 0.5, activation: str = "sigmoid"):
        """
        Args:
            threshold: Quantization threshold for binarizing activations.
            activation: Default activation function name.
        """
        self.threshold = threshold
        self.activation = activation
        self.stp = STPCore()

    def _activate(self, z: float, activation: Optional[str] = None) -> float:
        """Apply activation function."""
        act_name = activation or self.activation
        func = self.ACTIVATIONS.get(act_name)
        if func is None:
            raise ValueError(
                f"Unknown activation '{act_name}'. "
                f"Available: {list(self.ACTIVATIONS.keys())}"
            )
        return float(func(z))

    def _index_to_bits(self, j: int, n: int) -> List[int]:
        """
        Convert column index to input bit pattern.

        STP convention: j=0 → all-ones, j=2ⁿ-1 → all-zeros.
        """
        bits = []
        for i in range(n):
            bit = 1 - ((j >> (n - 1 - i)) & 1)
            bits.append(bit)
        return bits

    def build_neuron_matrix(
        self,
        weights: np.ndarray,
        bias: float,
        activation: Optional[str] = None,
        input_values: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Build the structure matrix for a single neuron.

        For a neuron with n inputs, the structure matrix M ∈ {0,1}^{2×2ⁿ}
        where column j encodes the boolean output for input combination j.

        Args:
            weights: Weight vector [w₁, ..., wₙ].
            bias: Bias term.
            activation: Activation function (overrides default).
            input_values: (low, high) values for boolean 0 and 1.
                          Default: (0.0, 1.0).

        Returns:
            2 × 2ⁿ structure matrix.
        """
        n = len(weights)
        n_states = 2**n
        M = np.zeros((2, n_states), dtype=np.float64)

        low, high = input_values or (0.0, 1.0)

        for j in range(n_states):
            bits = self._index_to_bits(j, n)
            # Map bits to actual input values
            x = np.array([high if b == 1 else low for b in bits])
            z = float(np.dot(weights, x)) + bias
            a = self._activate(z, activation)
            y = 1 if a >= self.threshold else 0
            M[:, j] = STPCore.bool_to_delta(y).flatten()

        return M

    def build_layer_matrix(
        self,
        weight_matrix: np.ndarray,
        bias_vector: np.ndarray,
        activation: Optional[str] = None,
        input_values: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Build the global structure matrix for an entire layer.

        For a layer with m output neurons and n inputs:
        - Each neuron yields Mᵢ ∈ {0,1}^{2×2ⁿ}
        - Layer matrix: L = M₁ * M₂ * ... * Mₘ (Khatri-Rao product)
        - Result: L ∈ {0,1}^{2ᵐ × 2ⁿ}

        Args:
            weight_matrix: Weight matrix W ∈ ℝ^{m×n}.
            bias_vector: Bias vector b ∈ ℝ^m.
            activation: Activation function.
            input_values: (low, high) for binary encoding.

        Returns:
            2ᵐ × 2ⁿ layer structure matrix.
        """
        m, n = weight_matrix.shape

        neuron_matrices = []
        for i in range(m):
            Mi = self.build_neuron_matrix(
                weights=weight_matrix[i, :],
                bias=float(bias_vector[i]),
                activation=activation,
                input_values=input_values,
            )
            neuron_matrices.append(Mi)

        # Compose via Khatri-Rao product
        L = neuron_matrices[0]
        for i in range(1, m):
            L = self.stp.khatri_rao_product(L, neuron_matrices[i])

        return L

    def build_network_matrix(
        self,
        layers_params: List[Tuple[np.ndarray, np.ndarray]],
        activation: Optional[str] = None,
    ) -> np.ndarray:
        """
        Build the global structure matrix for an entire network.

        L_total = L^(L) ⋉ L^(L-1) ⋉ ... ⋉ L^(1)

        Args:
            layers_params: List of (W, b) tuples for each layer.
            activation: Activation function (same for all layers).

        Returns:
            Global structure matrix (2^output_dim × 2^input_dim).
        """
        layer_matrices = []
        for i, (W, b) in enumerate(layers_params):
            L = self.build_layer_matrix(W, b, activation=activation)
            layer_matrices.append(L)

        # Compose: L_total = L_last @ ... @ L_first
        L_total = layer_matrices[-1]
        for L in reversed(layer_matrices[:-1]):
            L_total = self.stp.multiply(L_total, L)

        return L_total

    def build_from_pytorch(
        self,
        model,
        activation: Optional[str] = None,
    ) -> Dict:
        """
        Extract structure matrices from a PyTorch nn.Module.

        Automatically finds all nn.Linear layers and builds
        layer-wise and global structure matrices.

        Args:
            model: PyTorch model (nn.Module).
            activation: Activation function override.

        Returns:
            Dict with 'layer_matrices', 'global_matrix', 'layer_info'.
        """
        import torch.nn as nn

        layers_params = []
        layer_info = []

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.detach().cpu().numpy()
                b = module.bias.detach().cpu().numpy() if module.bias is not None else np.zeros(W.shape[0])
                layers_params.append((W, b))
                layer_info.append({
                    "name": name,
                    "in_features": W.shape[1],
                    "out_features": W.shape[0],
                })

        if not layers_params:
            raise ValueError("No nn.Linear layers found in model")

        layer_matrices = []
        for i, (W, b) in enumerate(layers_params):
            n_inputs = W.shape[1]
            if n_inputs > 20:
                print(
                    f"  Warning: Layer '{layer_info[i]['name']}' has {n_inputs} inputs. "
                    f"Structure matrix would require 2^{n_inputs} columns. "
                    f"Consider using LayerwiseAnalyzer or FeatureSpaceAnalyzer."
                )
                layer_matrices.append(None)
                continue
            L = self.build_layer_matrix(W, b, activation=activation)
            layer_matrices.append(L)

        # Build global matrix if all layers are tractable
        global_matrix = None
        if all(L is not None for L in layer_matrices):
            global_matrix = layer_matrices[-1]
            for L in reversed(layer_matrices[:-1]):
                global_matrix = self.stp.multiply(global_matrix, L)

        return {
            "layer_matrices": layer_matrices,
            "global_matrix": global_matrix,
            "layer_info": layer_info,
        }
