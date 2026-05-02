"""
SNA-STP Analyzer — Primary Interface
===================================

High-level API for SNA-STP (Structural Neural Analysis via Semi-tensor Product).

Provides three analysis modes:
- **Exact** (n ≤ 16): Complete structure matrix enumeration
- **Sample** (16 < n ≤ 100): Monte Carlo / importance sampling
- **Local** (any n): Neighborhood-based analysis around a reference input

Usage:
    >>> from snap_lib import SNAPAnalyzer
    >>> analyzer = SNAPAnalyzer(model, mode='exact')
    >>> result = analyzer.analyze()
    >>> print(result.logic_rules)
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from ..core.stp import STPCore
from ..core.structure_matrix import StructureMatrixBuilder
from ..core.logic_extractor import LogicExtractor


@dataclass
class SNAPResult:
    """Container for SNA-STP analysis results."""

    # Core outputs
    structure_matrix: Optional[np.ndarray] = None
    layer_matrices: List[np.ndarray] = field(default_factory=list)
    layer_info: List[Dict] = field(default_factory=list)

    # Logic rules
    logic_rules: Optional[Dict] = None
    truth_table: Optional[Dict] = None

    # Shapley values
    shapley_values: Optional[Dict[str, float]] = None

    # Feature importance
    feature_importance: Optional[List[Dict]] = None

    # Metadata
    mode: str = "exact"
    input_dim: int = 0
    output_dim: int = 0
    analysis_time: float = 0.0
    n_samples: int = 0

    def summary(self) -> str:
        """Human-readable summary of analysis results."""
        lines = [
            "=" * 60,
            f"SNA-STP Analysis Result ({self.mode} mode)",
            "=" * 60,
            f"Input dimension:  {self.input_dim}",
            f"Output dimension: {self.output_dim}",
            f"Analysis time:    {self.analysis_time:.3f}s",
        ]

        if self.structure_matrix is not None:
            lines.append(f"Structure matrix: {self.structure_matrix.shape}")

        if self.logic_rules:
            lines.append(f"\nLogic Rules (DNF):")
            lines.append(f"  {self.logic_rules.get('dnf', 'N/A')}")
            if self.logic_rules.get("gate_match"):
                lines.append(f"  Gate match: {self.logic_rules['gate_match']}")
            lines.append(
                f"  Activation ratio: {self.logic_rules.get('activation_ratio', 0):.1%}"
            )

        if self.shapley_values:
            lines.append(f"\nShapley Values:")
            for name, val in sorted(
                self.shapley_values.items(), key=lambda x: abs(x[1]), reverse=True
            ):
                lines.append(f"  {name}: {val:+.4f}")

        if self.feature_importance:
            lines.append(f"\nTop Features:")
            for fi in self.feature_importance[:5]:
                lines.append(
                    f"  #{fi['rank']} {fi['feature']}: influence={fi['influence']:.3f}"
                )

        lines.append("=" * 60)
        return "\n".join(lines)


class SNAPAnalyzer:
    """
    Primary SNA-STP analysis interface.

    Automatically selects the optimal analysis strategy based on
    network architecture and input dimension.

    Args:
        model: A PyTorch nn.Module. Must contain nn.Linear layers.
        mode: Analysis mode — 'auto', 'exact', 'sample', or 'local'.
        threshold: Quantization threshold for binarization.
        activation: Activation function name.
        var_names: Custom variable names for interpretability.

    Examples:
        >>> import torch.nn as nn
        >>> model = nn.Sequential(nn.Linear(2, 4), nn.Sigmoid(), nn.Linear(4, 1), nn.Sigmoid())
        >>> analyzer = SNAPAnalyzer(model)
        >>> result = analyzer.analyze()
        >>> print(result.summary())
    """

    MAX_EXACT_DIM = 16
    MAX_SAMPLE_DIM = 100

    def __init__(
        self,
        model=None,
        mode: str = "auto",
        threshold: float = 0.5,
        activation: str = "sigmoid",
        var_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.mode = mode
        self.threshold = threshold
        self.activation = activation
        self.var_names = var_names

        self.stp = STPCore()
        self.builder = StructureMatrixBuilder(threshold=threshold, activation=activation)
        self.extractor = LogicExtractor()

    def analyze(
        self,
        model=None,
        compute_shapley: bool = True,
        compute_importance: bool = True,
        top_k_features: int = 5,
    ) -> SNAPResult:
        """
        Run SNA-STP analysis on the model.

        Args:
            model: PyTorch model (overrides constructor model).
            compute_shapley: Whether to compute Shapley values.
            compute_importance: Whether to compute feature importance.
            top_k_features: Number of top features in importance ranking.

        Returns:
            SNAPResult containing all analysis outputs.
        """
        model = model or self.model
        if model is None:
            raise ValueError("No model provided. Pass model to constructor or analyze().")

        t_start = time.time()

        # Extract structure matrices from PyTorch model
        build_result = self.builder.build_from_pytorch(model, activation=self.activation)

        result = SNAPResult(
            layer_matrices=build_result["layer_matrices"],
            layer_info=build_result["layer_info"],
            structure_matrix=build_result["global_matrix"],
        )

        # Determine dimensions
        if build_result["layer_info"]:
            result.input_dim = build_result["layer_info"][0]["in_features"]
            result.output_dim = build_result["layer_info"][-1]["out_features"]

        # Determine mode
        mode = self._select_mode(result.input_dim) if self.mode == "auto" else self.mode
        result.mode = mode

        if mode == "exact" and result.structure_matrix is not None:
            var_names = self.var_names or [
                f"x{i+1}" for i in range(result.input_dim)
            ]

            # Extract logic rules
            result.logic_rules = self.extractor.extract_rules(
                result.structure_matrix, var_names=var_names
            )

            # Extract truth table
            result.truth_table = self.extractor.extract_truth_table(
                result.structure_matrix, var_names=var_names
            )

            # Compute Shapley values
            if compute_shapley and result.input_dim <= 12:
                result.shapley_values = self.extractor.compute_shapley_from_matrix(
                    result.structure_matrix, var_names=var_names
                )

            # Compute feature importance
            if compute_importance:
                result.feature_importance = self.extractor.extract_important_features(
                    result.structure_matrix,
                    var_names=var_names,
                    top_k=top_k_features,
                )

        elif mode == "sample":
            result = self._sample_analysis(model, result, top_k_features)

        elif mode == "local":
            result = self._local_analysis(model, result, top_k_features)

        result.analysis_time = time.time() - t_start
        return result

    def _select_mode(self, input_dim: int) -> str:
        """Auto-select the best analysis mode."""
        if input_dim <= self.MAX_EXACT_DIM:
            return "exact"
        elif input_dim <= self.MAX_SAMPLE_DIM:
            return "sample"
        else:
            return "local"

    def _sample_analysis(
        self, model, result: SNAPResult, top_k: int, n_samples: int = 10000
    ) -> SNAPResult:
        """Monte Carlo sampled analysis for medium-dimensional inputs."""
        import torch

        model.eval()
        n = result.input_dim

        # Generate random binary samples
        samples = np.random.randint(0, 2, size=(n_samples, n)).astype(np.float32)

        # Forward pass
        with torch.no_grad():
            X = torch.tensor(samples)
            outputs = model(X).numpy()

        # Binarize outputs
        if outputs.shape[1] == 1:
            preds = (outputs[:, 0] > self.threshold).astype(int)
        else:
            preds = outputs.argmax(axis=1)

        # Estimate feature importance via perturbation
        importance = np.zeros(n)
        base_pred_rate = preds.mean()

        for i in range(n):
            perturbed = samples.copy()
            perturbed[:, i] = 1 - perturbed[:, i]
            with torch.no_grad():
                X_p = torch.tensor(perturbed)
                out_p = model(X_p).numpy()
            if out_p.shape[1] == 1:
                preds_p = (out_p[:, 0] > self.threshold).astype(int)
            else:
                preds_p = out_p.argmax(axis=1)
            importance[i] = np.abs(preds - preds_p).mean()

        ranked = np.argsort(importance)[::-1][:top_k]
        result.feature_importance = [
            {
                "feature": f"x{idx+1}",
                "feature_index": int(idx),
                "influence": float(importance[idx]),
                "rank": rank + 1,
            }
            for rank, idx in enumerate(ranked)
        ]
        result.n_samples = n_samples
        result.mode = "sample"

        return result

    def _local_analysis(
        self,
        model,
        result: SNAPResult,
        top_k: int,
        reference: Optional[List[int]] = None,
        radius: int = 3,
    ) -> SNAPResult:
        """Local neighborhood analysis for high-dimensional inputs."""
        import torch
        from itertools import combinations

        model.eval()
        n = result.input_dim

        if reference is None:
            reference = [0] * n  # Default reference: all zeros

        ref_array = np.array(reference, dtype=np.float32)

        # Analyze local neighborhood within Hamming distance r
        local_table = []

        for r in range(radius + 1):
            for flip_positions in combinations(range(n), r):
                x = ref_array.copy()
                for pos in flip_positions:
                    x[pos] = 1 - x[pos]

                with torch.no_grad():
                    out = model(torch.tensor(x.reshape(1, -1))).numpy()

                if out.shape[1] == 1:
                    pred = 1 if out[0, 0] > self.threshold else 0
                else:
                    pred = int(out.argmax())

                local_table.append({
                    "input": x.astype(int).tolist(),
                    "output": pred,
                    "distance": r,
                    "flipped": list(flip_positions),
                })

        # Estimate local feature importance
        importance = np.zeros(n)
        ref_out = local_table[0]["output"]

        for entry in local_table:
            if entry["distance"] == 1 and entry["output"] != ref_out:
                importance[entry["flipped"][0]] += 1

        n_single = sum(1 for e in local_table if e["distance"] == 1)
        if n_single > 0:
            importance /= max(1, n_single / n)

        ranked = np.argsort(importance)[::-1][:top_k]
        result.feature_importance = [
            {
                "feature": f"x{idx+1}",
                "feature_index": int(idx),
                "influence": float(importance[idx]),
                "rank": rank + 1,
            }
            for rank, idx in enumerate(ranked)
        ]
        result.n_samples = len(local_table)
        result.mode = "local"

        return result

    def trace_trajectory(
        self, input_bits: List[int]
    ) -> List[Dict]:
        """
        Trace the logical trajectory of an input through all layers.

        Args:
            input_bits: Binary input vector.

        Returns:
            List of dicts describing state at each layer.
        """
        if not any(L is not None for L in self._get_layer_matrices()):
            raise ValueError("Layer matrices not available. Run analyze() first.")

        trajectory = []
        X = self.stp.encode_state(input_bits)
        trajectory.append({
            "layer": 0,
            "name": "input",
            "state": self.stp.decode_state(X),
            "state_index": self.stp.state_to_index(X),
        })

        for i, L in enumerate(self._get_layer_matrices()):
            if L is None:
                break
            X = L @ X
            n_out = int(np.log2(L.shape[0]))
            trajectory.append({
                "layer": i + 1,
                "name": f"layer_{i+1}",
                "state": self.stp.decode_state(X) if X.shape[0] <= 1024 else None,
                "state_index": self.stp.state_to_index(X),
                "n_neurons": n_out,
            })

        return trajectory

    def _get_layer_matrices(self) -> List:
        """Get layer matrices from last analysis."""
        # This is a helper; in practice, store from last analyze() call.
        if self.model is None:
            return []
        result = self.builder.build_from_pytorch(self.model, self.activation)
        return result["layer_matrices"]

    def verify(
        self,
        test_function,
        input_dim: Optional[int] = None,
    ) -> Dict:
        """
        Verify extracted logic against a known boolean function.

        Args:
            test_function: Callable that takes *bits and returns 0 or 1.
            input_dim: Number of input bits. Auto-detected if possible.

        Returns:
            Dict with 'accuracy', 'correct', 'total', 'errors'.
        """
        if self.model is None:
            raise ValueError("No model set")

        result = self.builder.build_from_pytorch(self.model, self.activation)
        M = result["global_matrix"]

        if M is None:
            raise ValueError("Global matrix unavailable (dimensions too large)")

        n = int(np.log2(M.shape[1]))
        correct = 0
        errors = []

        for j in range(2**n):
            bits = []
            for i in range(n):
                bits.append(1 - ((j >> (n - 1 - i)) & 1))

            X = self.stp.encode_state(bits)
            Y = M @ X
            stp_out = self.stp.delta_to_bool(Y)
            expected = test_function(*bits)

            if stp_out == expected:
                correct += 1
            else:
                errors.append({"input": bits, "stp": stp_out, "expected": expected})

        return {
            "accuracy": correct / 2**n,
            "correct": correct,
            "total": 2**n,
            "errors": errors,
        }
