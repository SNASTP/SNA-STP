"""
Layerwise Analyzer
==================

Analyzes neural networks layer by layer, avoiding the exponential
explosion of global structure matrix computation.

Complexity: O(L × 2^d_max) instead of O(2^n_input), where
L = number of layers, d_max = max layer width.

Best for: Deep networks with narrow bottleneck layers.
"""

import numpy as np
from typing import Dict, List, Optional
from ..core.stp import STPCore
from ..core.structure_matrix import StructureMatrixBuilder
from ..core.logic_extractor import LogicExtractor


class LayerwiseAnalyzer:
    """
    Layer-by-layer STP analysis.

    Instead of computing one global structure matrix (which requires
    O(2^n) columns for n input features), this analyzer computes
    smaller structure matrices for each layer independently.

    This enables analysis of much deeper networks while maintaining
    interpretability at each layer.

    Examples:
        >>> analyzer = LayerwiseAnalyzer(threshold=0.5)
        >>> model = nn.Sequential(nn.Linear(4, 8), nn.Sigmoid(), nn.Linear(8, 2), nn.Sigmoid())
        >>> results = analyzer.analyze_model(model)
        >>> print(results['summary'])
    """

    def __init__(self, threshold: float = 0.5, activation: str = "sigmoid"):
        self.threshold = threshold
        self.activation = activation
        self.builder = StructureMatrixBuilder(threshold=threshold, activation=activation)
        self.extractor = LogicExtractor()
        self.stp = STPCore()

    def analyze_layer(
        self,
        weight: np.ndarray,
        bias: np.ndarray,
        layer_name: str = "",
        max_enum: int = 1024,
    ) -> Dict:
        """
        Analyze a single layer's logical behavior.

        For each neuron, determines:
        - Activation ratio (fraction of inputs that activate it)
        - Most important input dimensions
        - Logic pattern (if input dim is small enough for exact analysis)

        Args:
            weight: Weight matrix (m × n).
            bias: Bias vector (m,).
            layer_name: Human-readable name for the layer.
            max_enum: Maximum number of input patterns to enumerate.

        Returns:
            Dict with per-neuron analysis.
        """
        m, n = weight.shape

        neuron_analyses = []
        for i in range(m):
            w = weight[i, :]
            b = bias[i]

            # Enumerate input patterns (up to max_enum)
            n_patterns = min(2**n, max_enum)
            activating = 0

            for j in range(n_patterns):
                bits = self._index_to_bits(j, n)
                z = float(np.dot(w, bits)) + b
                a = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
                if a >= self.threshold:
                    activating += 1

            # Feature importance via weight magnitude
            importance = np.abs(w)
            top_indices = np.argsort(importance)[-min(3, n):][::-1].tolist()

            neuron_analyses.append({
                "neuron_id": i,
                "activation_ratio": activating / n_patterns,
                "important_dims": top_indices,
                "weight_magnitude": float(np.abs(w).mean()),
                "bias": float(b),
                "weight_norm": float(np.linalg.norm(w)),
            })

        # Build structure matrix if feasible
        structure_matrix = None
        logic_rules = None
        if n <= 16:
            structure_matrix = self.builder.build_layer_matrix(weight, bias, self.activation)
            if m == 1:
                logic_rules = self.extractor.extract_rules(structure_matrix)

        return {
            "layer_name": layer_name,
            "input_dim": n,
            "output_dim": m,
            "neurons": neuron_analyses,
            "structure_matrix": structure_matrix,
            "logic_rules": logic_rules,
        }

    def analyze_model(self, model, layer_names: Optional[List[str]] = None) -> Dict:
        """
        Analyze all linear layers of a PyTorch model.

        Args:
            model: PyTorch nn.Module.
            layer_names: Optional custom names for each layer.

        Returns:
            Dict with per-layer analyses and summary.
        """
        import torch.nn as nn

        layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.detach().cpu().numpy()
                b = module.bias.detach().cpu().numpy() if module.bias is not None else np.zeros(W.shape[0])
                layers.append((name, W, b))

        results = []
        for i, (name, W, b) in enumerate(layers):
            lname = layer_names[i] if layer_names and i < len(layer_names) else name
            analysis = self.analyze_layer(W, b, layer_name=lname)
            results.append(analysis)

        return {
            "layers": results,
            "n_layers": len(results),
            "summary": self._generate_summary(results),
        }

    def _generate_summary(self, layer_results: List[Dict]) -> str:
        """Generate a human-readable summary."""
        lines = ["=" * 50, "Layerwise SNA-STP Analysis Summary", "=" * 50]

        for lr in layer_results:
            lines.append(f"\n--- {lr['layer_name']} ({lr['input_dim']}→{lr['output_dim']}) ---")
            for neuron in lr["neurons"]:
                lines.append(
                    f"  Neuron {neuron['neuron_id']}: "
                    f"act_ratio={neuron['activation_ratio']:.1%}, "
                    f"top_dims={neuron['important_dims']}"
                )
            if lr["logic_rules"]:
                lines.append(f"  Logic: {lr['logic_rules'].get('dnf', 'N/A')}")

        return "\n".join(lines)

    def _index_to_bits(self, j: int, n: int) -> List[int]:
        bits = []
        for i in range(n):
            bit = 1 - ((j >> (n - 1 - i)) & 1)
            bits.append(bit)
        return bits
