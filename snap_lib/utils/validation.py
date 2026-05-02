"""
Validation Utilities
====================

Tools for validating SNA-STP analysis correctness.
"""

import numpy as np
import time
from typing import Dict, Optional, Callable, List
from ..core.stp import STPCore


def validate_consistency(
    model,
    structure_matrix: np.ndarray,
    n_inputs: int,
    threshold: float = 0.5,
) -> Dict:
    """
    Validate STP-NN consistency by comparing neural network predictions
    with structure matrix predictions across all input combinations.

    Args:
        model: PyTorch model.
        structure_matrix: Global structure matrix from SNA-STP.
        n_inputs: Number of input features.
        threshold: Binarization threshold.

    Returns:
        Dict with consistency ratio, error details, timing info.
    """
    import torch

    stp = STPCore()
    model.eval()

    total = 0
    consistent = 0
    errors = []

    t_start = time.time()

    with torch.no_grad():
        for j in range(2**n_inputs):
            bits = []
            for i in range(n_inputs):
                bits.append(1 - ((j >> (n_inputs - 1 - i)) & 1))

            # Neural network prediction
            x = torch.tensor([bits], dtype=torch.float32)
            nn_out = model(x).numpy()
            if nn_out.shape[1] == 1:
                nn_pred = 1 if nn_out[0, 0] > threshold else 0
            else:
                nn_pred = int(nn_out.argmax())

            # STP prediction
            X = stp.encode_state(bits)
            Y = structure_matrix @ X
            stp_pred = stp.delta_to_bool(Y)

            total += 1
            if nn_pred == stp_pred:
                consistent += 1
            else:
                errors.append({
                    "input": bits,
                    "nn_pred": nn_pred,
                    "stp_pred": stp_pred,
                    "nn_raw": float(nn_out[0, 0]) if nn_out.shape[1] == 1 else nn_out[0].tolist(),
                })

    elapsed = time.time() - t_start

    return {
        "consistency": consistent / total,
        "consistent": consistent,
        "total": total,
        "errors": errors,
        "n_errors": len(errors),
        "elapsed": elapsed,
    }


def validate_against_function(
    structure_matrix: np.ndarray,
    truth_function: Callable,
    n_inputs: int,
) -> Dict:
    """
    Validate a structure matrix against a known boolean function.

    Args:
        structure_matrix: The matrix to validate.
        truth_function: f(*bits) → 0 or 1.
        n_inputs: Number of inputs.

    Returns:
        Dict with accuracy, errors.
    """
    stp = STPCore()
    correct = 0
    errors = []

    for j in range(2**n_inputs):
        bits = []
        for i in range(n_inputs):
            bits.append(1 - ((j >> (n_inputs - 1 - i)) & 1))

        X = stp.encode_state(bits)
        Y = structure_matrix @ X
        stp_out = stp.delta_to_bool(Y)
        expected = truth_function(*bits)

        if stp_out == expected:
            correct += 1
        else:
            errors.append({"input": bits, "got": stp_out, "expected": expected})

    return {
        "accuracy": correct / 2**n_inputs,
        "correct": correct,
        "total": 2**n_inputs,
        "errors": errors,
    }


def benchmark_complexity(
    max_n: int = 14,
    step: int = 2,
) -> List[Dict]:
    """
    Benchmark structure matrix construction time for increasing input dimensions.

    Args:
        max_n: Maximum number of inputs to test.
        step: Step size for n.

    Returns:
        List of dicts with n, time, matrix_size.
    """
    from ..core.structure_matrix import StructureMatrixBuilder

    builder = StructureMatrixBuilder()
    results = []

    for n in range(2, max_n + 1, step):
        W = np.random.randn(1, n)
        b = np.random.randn(1)

        t_start = time.time()
        M = builder.build_neuron_matrix(W[0], float(b[0]))
        elapsed = time.time() - t_start

        results.append({
            "n_inputs": n,
            "time_seconds": elapsed,
            "matrix_columns": 2**n,
            "matrix_shape": M.shape,
        })

        print(f"  n={n:2d}: {2**n:>8} columns, {elapsed:.4f}s")

    return results
