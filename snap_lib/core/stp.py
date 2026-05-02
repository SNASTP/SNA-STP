"""
Semi-Tensor Product (STP) Core Operations
==========================================

Implements the fundamental STP algebra:
  A ⋉ B = (A ⊗ I_{t/n})(B ⊗ I_{t/p})
  where t = lcm(n, p)

When n == p, STP degenerates to standard matrix multiplication.

References:
    Cheng, D. (2011). "Semi-tensor product of matrices and its applications
    to Morgan's problem." Science China Information Sciences.
"""

import numpy as np
from typing import List, Optional, Tuple
from math import gcd


class STPCore:
    """
    Semi-Tensor Product core computation engine.

    Provides all fundamental STP operations needed for neural network
    structural analysis, including:
    - Boolean-to-delta vector mapping
    - Semi-tensor product multiplication
    - Khatri-Rao product for layer composition
    - State encoding/decoding

    Examples:
        >>> stp = STPCore()
        >>> # Boolean to delta vector
        >>> stp.bool_to_delta(1)
        array([[1], [0]])
        >>> # Multi-variable state
        >>> stp.encode_state([1, 0, 1])
        array([[0], [0], [1], [0], [0], [0], [0], [0]])
        >>> # Semi-tensor product
        >>> A = np.array([[1, 0, 0, 0], [0, 1, 1, 1]])
        >>> x = stp.encode_state([1, 1])
        >>> stp.multiply(A, x)
        array([[1], [0]])
    """

    # Standard logic gate structure matrices
    LOGIC_GATES = {
        "NOT": np.array([[0, 1], [1, 0]]),
        "AND": np.array([[1, 0, 0, 0], [0, 1, 1, 1]]),
        "OR": np.array([[1, 1, 1, 0], [0, 0, 0, 1]]),
        "XOR": np.array([[0, 1, 1, 0], [1, 0, 0, 1]]),
        "NAND": np.array([[0, 1, 1, 1], [1, 0, 0, 0]]),
        "NOR": np.array([[0, 0, 0, 1], [1, 1, 1, 0]]),
        "XNOR": np.array([[1, 0, 0, 1], [0, 1, 1, 0]]),
        "IDENTITY": np.array([[1, 0], [0, 1]]),
    }

    @staticmethod
    def bool_to_delta(b: int) -> np.ndarray:
        """
        Map a scalar boolean to a delta (canonical) vector.

        φ: {0, 1} → {δ₂¹, δ₂²}
        1 → [1, 0]ᵀ (δ₂¹)
        0 → [0, 1]ᵀ (δ₂²)

        Args:
            b: Boolean value (0 or 1).

        Returns:
            2×1 column vector.
        """
        if b == 1:
            return np.array([[1], [0]], dtype=np.float64)
        else:
            return np.array([[0], [1]], dtype=np.float64)

    @staticmethod
    def delta_to_bool(delta: np.ndarray) -> int:
        """
        Map a delta vector back to a scalar boolean.

        Args:
            delta: 2×1 column vector (or any array with first two elements).

        Returns:
            1 if first element > second, else 0.
        """
        flat = delta.flatten()
        return 1 if flat[0] > flat[1] else 0

    @staticmethod
    def encode_state(bits: List[int]) -> np.ndarray:
        """
        Multi-variable joint encoding via iterated Kronecker product.

        Φ: {0,1}ⁿ → Δ_{2ⁿ}

        Args:
            bits: List of boolean values [b₁, b₂, ..., bₙ].

        Returns:
            2ⁿ × 1 canonical basis vector.
        """
        state = STPCore.bool_to_delta(bits[0])
        for b in bits[1:]:
            state = np.kron(state, STPCore.bool_to_delta(b))
        return state

    @staticmethod
    def decode_state(state: np.ndarray) -> List[int]:
        """
        Decode a state vector back to a bit list.

        Args:
            state: 2ⁿ × 1 canonical basis vector.

        Returns:
            List of n boolean values.
        """
        idx = int(np.argmax(state.flatten()))
        n = int(np.log2(len(state.flatten())))
        bits = []
        for i in range(n):
            bits.append(1 - ((idx >> (n - 1 - i)) & 1))
        return bits

    @staticmethod
    def state_to_index(state: np.ndarray) -> int:
        """
        Convert a state vector to its 1-based index.

        δ_{2ⁿ}^j → j

        Args:
            state: Canonical basis vector.

        Returns:
            1-based index of the active element.
        """
        return int(np.argmax(state.flatten())) + 1

    @staticmethod
    def index_to_state(j: int, n: int) -> np.ndarray:
        """
        Convert a 1-based index to a state vector.

        j → δ_{2ⁿ}^j

        Args:
            j: 1-based index.
            n: Number of boolean variables.

        Returns:
            2ⁿ × 1 canonical basis vector.
        """
        state = np.zeros((2**n, 1), dtype=np.float64)
        state[j - 1, 0] = 1.0
        return state

    @staticmethod
    def multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Semi-tensor product: A ⋉ B.

        Handles dimension mismatch through Kronecker product extension:
        - If cols(A) == rows(B): standard matrix multiplication
        - If cols(A) > rows(B) and cols(A) % rows(B) == 0:
          B_ext = B ⊗ I_{cols(A)/rows(B)}, then A @ B_ext
        - If rows(B) > cols(A) and rows(B) % cols(A) == 0:
          A_ext = A ⊗ I_{rows(B)/cols(A)}, then A_ext @ B

        Args:
            A: Left matrix (m × n).
            B: Right matrix (p × q).

        Returns:
            Result of A ⋉ B.

        Raises:
            ValueError: If dimensions are incompatible (no integer ratio).
        """
        m, n = A.shape
        p = B.shape[0]
        q = B.shape[1] if B.ndim > 1 else 1

        if n == p:
            return A @ B
        elif n > p and n % p == 0:
            k = n // p
            B_ext = np.kron(B, np.eye(k))
            return A @ B_ext
        elif p > n and p % n == 0:
            k = p // n
            A_ext = np.kron(A, np.eye(k))
            return A_ext @ B
        else:
            # General case: use lcm
            t = (n * p) // gcd(n, p)
            A_ext = np.kron(A, np.eye(t // n))
            B_ext = np.kron(B, np.eye(t // p))
            return A_ext @ B_ext

    @staticmethod
    def khatri_rao_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Khatri-Rao product (column-wise Kronecker product).

        Used to compose multiple neuron structure matrices into a
        layer-level global structure matrix.

        [A * B]_{:,j} = A_{:,j} ⊗ B_{:,j}

        Args:
            A: Matrix of shape (m₁, k).
            B: Matrix of shape (m₂, k).

        Returns:
            Matrix of shape (m₁·m₂, k).

        Raises:
            ValueError: If A and B have different number of columns.
        """
        if A.shape[1] != B.shape[1]:
            raise ValueError(
                f"Column count mismatch: A has {A.shape[1]}, B has {B.shape[1]}"
            )
        k = A.shape[1]
        result = np.zeros((A.shape[0] * B.shape[0], k), dtype=np.float64)
        for j in range(k):
            result[:, j] = np.kron(A[:, j], B[:, j])
        return result

    @classmethod
    def gate_matrix(cls, gate_name: str) -> np.ndarray:
        """
        Get the structure matrix for a standard logic gate.

        Args:
            gate_name: One of 'NOT', 'AND', 'OR', 'XOR', 'NAND', 'NOR', 'XNOR', 'IDENTITY'.

        Returns:
            Structure matrix for the gate.
        """
        name = gate_name.upper()
        if name not in cls.LOGIC_GATES:
            raise ValueError(
                f"Unknown gate '{gate_name}'. Available: {list(cls.LOGIC_GATES.keys())}"
            )
        return cls.LOGIC_GATES[name].copy().astype(np.float64)

    @staticmethod
    def hamming_distance(M1: np.ndarray, M2: np.ndarray) -> int:
        """
        Compute the Hamming distance between two binary structure matrices.

        Args:
            M1, M2: Binary matrices of the same shape.

        Returns:
            Number of differing entries.
        """
        return int(np.sum(np.abs(M1 - M2) > 0.5))

    @staticmethod
    def match_logic_gate(M: np.ndarray, threshold: float = 0.0) -> Optional[str]:
        """
        Identify the closest standard logic gate by Hamming distance.

        Args:
            M: A 2×4 structure matrix.
            threshold: Maximum Hamming distance to accept (0 = exact match only).

        Returns:
            Gate name if matched, None otherwise.
        """
        if M.shape != (2, 4):
            return None

        best_gate = None
        best_dist = float("inf")

        for name, gate_M in STPCore.LOGIC_GATES.items():
            if gate_M.shape != (2, 4):
                continue
            dist = STPCore.hamming_distance(M, gate_M)
            if dist < best_dist:
                best_dist = dist
                best_gate = name

        if best_dist <= threshold:
            return best_gate
        if best_dist == 0:
            return best_gate
        return None
