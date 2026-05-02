"""
Logic Rule Extractor
====================

Extracts human-readable boolean logic rules from structure matrices.

Given a structure matrix M, this module can:
- Generate Disjunctive Normal Form (DNF) expressions
- Generate Conjunctive Normal Form (CNF) expressions
- Identify standard logic gates
- Simplify boolean expressions
- Extract truth tables
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from .stp import STPCore


class LogicExtractor:
    """
    Extract and interpret boolean logic rules from structure matrices.

    Examples:
        >>> extractor = LogicExtractor()
        >>> M_xor = np.array([[0, 1, 1, 0], [1, 0, 0, 1]])
        >>> rules = extractor.extract_rules(M_xor, var_names=['A', 'B'])
        >>> print(rules['dnf'])
        '(¬A ∧ B) ∨ (A ∧ ¬B)'
    """

    def __init__(self, var_prefix: str = "x"):
        """
        Args:
            var_prefix: Prefix for variable names (e.g., 'x' → x1, x2, ...).
        """
        self.var_prefix = var_prefix
        self.stp = STPCore()

    def _index_to_bits(self, j: int, n: int) -> List[int]:
        """Convert column index to bit pattern (STP convention)."""
        bits = []
        for i in range(n):
            bit = 1 - ((j >> (n - 1 - i)) & 1)
            bits.append(bit)
        return bits

    def extract_truth_table(
        self, M: np.ndarray, var_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Extract the complete truth table from a structure matrix.

        Args:
            M: Structure matrix (2 × 2ⁿ for single-output, 2ᵐ × 2ⁿ for multi-output).
            var_names: Variable names. Auto-generated if None.

        Returns:
            Dict with 'inputs', 'outputs', 'table' (list of dicts).
        """
        n_cols = M.shape[1]
        n_inputs = int(np.log2(n_cols))
        n_outputs = int(np.log2(M.shape[0])) if M.shape[0] > 2 else 1

        if var_names is None:
            var_names = [f"{self.var_prefix}{i+1}" for i in range(n_inputs)]

        table = []
        for j in range(n_cols):
            bits = self._index_to_bits(j, n_inputs)

            if n_outputs == 1:
                output = self.stp.delta_to_bool(M[:, j:j+1])
                outputs = [output]
            else:
                state_idx = int(np.argmax(M[:, j]))
                outputs = []
                for k in range(n_outputs):
                    outputs.append(1 - ((state_idx >> (n_outputs - 1 - k)) & 1))

            table.append({
                "inputs": dict(zip(var_names, bits)),
                "outputs": outputs,
                "input_bits": bits,
            })

        return {
            "n_inputs": n_inputs,
            "n_outputs": n_outputs,
            "var_names": var_names,
            "table": table,
        }

    def extract_rules(
        self,
        M: np.ndarray,
        var_names: Optional[List[str]] = None,
        output_index: int = 0,
    ) -> Dict:
        """
        Extract boolean logic rules from a structure matrix.

        Returns DNF (Disjunctive Normal Form) and CNF (Conjunctive Normal Form)
        representations.

        Args:
            M: Structure matrix.
            var_names: Variable names.
            output_index: Which output to extract rules for (for multi-output).

        Returns:
            Dict with 'dnf', 'cnf', 'true_patterns', 'false_patterns',
            'gate_match', 'activation_ratio'.
        """
        tt = self.extract_truth_table(M, var_names)
        n_inputs = tt["n_inputs"]
        names = tt["var_names"]

        true_patterns = []
        false_patterns = []

        for row in tt["table"]:
            if row["outputs"][output_index] == 1:
                true_patterns.append(row["input_bits"])
            else:
                false_patterns.append(row["input_bits"])

        # Generate DNF
        dnf = self._patterns_to_dnf(true_patterns, names)

        # Generate CNF
        cnf = self._patterns_to_cnf(false_patterns, names)

        # Try to match a standard gate
        gate_match = STPCore.match_logic_gate(M) if M.shape == (2, 4) else None

        return {
            "dnf": dnf,
            "cnf": cnf,
            "true_patterns": true_patterns,
            "false_patterns": false_patterns,
            "gate_match": gate_match,
            "activation_ratio": len(true_patterns) / (len(true_patterns) + len(false_patterns)),
            "n_inputs": n_inputs,
            "var_names": names,
        }

    def _patterns_to_dnf(
        self, patterns: List[List[int]], var_names: List[str]
    ) -> str:
        """Convert true-patterns to Disjunctive Normal Form string."""
        if len(patterns) == 0:
            return "FALSE"
        n = len(var_names)
        if len(patterns) == 2**n:
            return "TRUE"

        terms = []
        for pattern in patterns:
            literals = []
            for i, b in enumerate(pattern):
                if b == 1:
                    literals.append(var_names[i])
                else:
                    literals.append(f"¬{var_names[i]}")
            terms.append("(" + " ∧ ".join(literals) + ")")

        return " ∨ ".join(terms)

    def _patterns_to_cnf(
        self, false_patterns: List[List[int]], var_names: List[str]
    ) -> str:
        """Convert false-patterns to Conjunctive Normal Form string."""
        if len(false_patterns) == 0:
            return "TRUE"
        n = len(var_names)
        if len(false_patterns) == 2**n:
            return "FALSE"

        clauses = []
        for pattern in false_patterns:
            # Each false pattern becomes a clause (disjunction of negated literals)
            literals = []
            for i, b in enumerate(pattern):
                if b == 1:
                    literals.append(f"¬{var_names[i]}")
                else:
                    literals.append(var_names[i])
            clauses.append("(" + " ∨ ".join(literals) + ")")

        return " ∧ ".join(clauses)

    def extract_important_features(
        self, M: np.ndarray, var_names: Optional[List[str]] = None, top_k: int = 5
    ) -> List[Dict]:
        """
        Identify the most influential input features based on
        how much flipping each feature changes the output.

        Args:
            M: Structure matrix (2 × 2ⁿ).
            var_names: Variable names.
            top_k: Number of top features to return.

        Returns:
            List of dicts with 'feature', 'influence', 'rank'.
        """
        n_cols = M.shape[1]
        n = int(np.log2(n_cols))

        if var_names is None:
            var_names = [f"{self.var_prefix}{i+1}" for i in range(n)]

        # Compute influence: how often flipping variable i changes the output
        influence = np.zeros(n)

        for j in range(n_cols):
            bits = self._index_to_bits(j, n)
            out_j = 1 if M[0, j] > 0.5 else 0

            for k in range(n):
                flipped = bits.copy()
                flipped[k] = 1 - flipped[k]
                # Find column index for flipped input
                flipped_j = 0
                for i, b in enumerate(flipped):
                    flipped_j |= ((1 - b) << (n - 1 - i))
                out_f = 1 if M[0, flipped_j] > 0.5 else 0

                if out_j != out_f:
                    influence[k] += 1

        influence /= n_cols  # Normalize

        # Rank features
        ranked = np.argsort(influence)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(ranked):
            results.append({
                "feature": var_names[idx],
                "feature_index": int(idx),
                "influence": float(influence[idx]),
                "rank": rank + 1,
            })

        return results

    def compute_shapley_from_matrix(
        self, M: np.ndarray, var_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute exact Shapley values directly from the structure matrix.

        This implements the SNA-STP→SHAP conversion theorem:
        φᵢ = Σ_{S⊆N\\{i}} [|S|!(n-|S|-1)!/n!] × [v(S∪{i}) - v(S)]

        Args:
            M: Structure matrix (2 × 2ⁿ).
            var_names: Variable names.

        Returns:
            Dict mapping variable names to Shapley values.
        """
        from math import factorial

        n_cols = M.shape[1]
        n = int(np.log2(n_cols))

        if var_names is None:
            var_names = [f"{self.var_prefix}{i+1}" for i in range(n)]

        # Build value function v(S) from truth table
        # v(S) = expected output when features in S are set to 1, others marginalized
        def v(subset_mask: int) -> float:
            """Value function: average output over all completions of the subset."""
            total = 0.0
            count = 0
            # Iterate over all possible values for features NOT in subset
            free_positions = [k for k in range(n) if not (subset_mask & (1 << k))]
            n_free = len(free_positions)

            for code in range(2**n_free):
                bits = [0] * n
                # Set subset features to 1
                for k in range(n):
                    if subset_mask & (1 << k):
                        bits[k] = 1
                # Set free features according to code
                for idx, pos in enumerate(free_positions):
                    bits[pos] = (code >> idx) & 1

                # Look up in structure matrix
                j = 0
                for i, b in enumerate(bits):
                    j |= ((1 - b) << (n - 1 - i))
                total += 1 if M[0, j] > 0.5 else 0
                count += 1

            return total / count if count > 0 else 0.0

        # Compute Shapley values
        shapley = {}
        for i in range(n):
            phi_i = 0.0
            # Iterate over all subsets S ⊆ N\{i}
            others = [k for k in range(n) if k != i]
            n_others = len(others)

            for s_code in range(2**n_others):
                # Build subset mask (without feature i)
                s_mask = 0
                s_size = 0
                for idx, k in enumerate(others):
                    if s_code & (1 << idx):
                        s_mask |= (1 << k)
                        s_size += 1

                # v(S ∪ {i}) - v(S)
                s_with_i = s_mask | (1 << i)
                marginal = v(s_with_i) - v(s_mask)

                # Shapley weight
                weight = factorial(s_size) * factorial(n - s_size - 1) / factorial(n)
                phi_i += weight * marginal

            shapley[var_names[i]] = phi_i

        return shapley

    def compute_instance_shapley(
        self,
        M: np.ndarray,
        x_instance: List[int],
        var_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute per-instance Shapley values from the structure matrix.

        Unlike compute_shapley_from_matrix (global Shapley where v(S)
        marginalizes by setting S-features to 1), this computes
        LOCAL Shapley values for a specific instance x:

            v_x(S) = E_{z~Unif}[ f(x_S, z_{N\\S}) ]

        Features in S take their actual values from x_instance,
        features not in S are marginalized over {0, 1} uniformly.

        This is EXACTLY what KernelSHAP approximates, so comparing
        SNA-STP instance Shapley with KernelSHAP is mathematically valid.

        Args:
            M: Structure matrix (2 × 2^n).
            x_instance: Input bit vector [b_1, ..., b_n].
            var_names: Variable names.

        Returns:
            Dict mapping variable names to per-instance Shapley values.
        """
        from math import factorial

        n_cols = M.shape[1]
        n = int(np.log2(n_cols))
        x = list(x_instance)

        if var_names is None:
            var_names = [f"{self.var_prefix}{i+1}" for i in range(n)]

        def v_x(subset_mask: int) -> float:
            """Value function: features in S take values from x, others marginalized."""
            free_positions = [k for k in range(n) if not (subset_mask & (1 << k))]
            n_free = len(free_positions)
            total = 0.0

            for code in range(2 ** n_free):
                bits = [0] * n
                # Features in S: use their values from x_instance
                for k in range(n):
                    if subset_mask & (1 << k):
                        bits[k] = x[k]
                # Free features: enumerate
                for idx, pos in enumerate(free_positions):
                    bits[pos] = (code >> idx) & 1

                # Look up in structure matrix
                j = 0
                for i, b in enumerate(bits):
                    j |= ((1 - b) << (n - 1 - i))
                total += 1 if M[0, j] > 0.5 else 0

            return total / (2 ** n_free) if n_free > 0 else total

        # Compute Shapley values
        shapley = {}
        for i in range(n):
            phi_i = 0.0
            others = [k for k in range(n) if k != i]
            n_others = len(others)

            for s_code in range(2 ** n_others):
                s_mask = 0
                s_size = 0
                for idx, k in enumerate(others):
                    if s_code & (1 << idx):
                        s_mask |= (1 << k)
                        s_size += 1

                s_with_i = s_mask | (1 << i)
                marginal = v_x(s_with_i) - v_x(s_mask)

                weight = factorial(s_size) * factorial(n - s_size - 1) / factorial(n)
                phi_i += weight * marginal

            shapley[var_names[i]] = phi_i

        return shapley

    def counterfactual_query(
        self,
        M: np.ndarray,
        original_input: List[int],
        target_output: int = 1,
        var_names: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Find minimal changes to flip the output.

        Given an input and a desired target output, find all minimal
        sets of feature flips that achieve the target.

        Args:
            M: Structure matrix (2 × 2ⁿ).
            original_input: Current input bit vector.
            target_output: Desired output (0 or 1).
            var_names: Variable names.

        Returns:
            List of counterfactual explanations sorted by number of changes.
        """
        n = len(original_input)
        if var_names is None:
            var_names = [f"{self.var_prefix}{i+1}" for i in range(n)]

        # Current output
        j_orig = 0
        for i, b in enumerate(original_input):
            j_orig |= ((1 - b) << (n - 1 - i))
        current_output = 1 if M[0, j_orig] > 0.5 else 0

        if current_output == target_output:
            return [{"changes": [], "message": "Already at target output"}]

        results = []

        # Search by increasing number of flips
        for n_flips in range(1, n + 1):
            found = self._find_flips(
                M, original_input, target_output, n_flips, var_names
            )
            results.extend(found)
            if results:
                break  # Only return minimal flips

        return results

    def _find_flips(
        self,
        M: np.ndarray,
        original: List[int],
        target: int,
        n_flips: int,
        var_names: List[str],
    ) -> List[Dict]:
        """Find all combinations of n_flips that achieve target output."""
        from itertools import combinations

        n = len(original)
        results = []

        for flip_indices in combinations(range(n), n_flips):
            new_input = original.copy()
            for idx in flip_indices:
                new_input[idx] = 1 - new_input[idx]

            j = 0
            for i, b in enumerate(new_input):
                j |= ((1 - b) << (n - 1 - i))
            output = 1 if M[0, j] > 0.5 else 0

            if output == target:
                changes = []
                for idx in flip_indices:
                    changes.append({
                        "feature": var_names[idx],
                        "from": original[idx],
                        "to": new_input[idx],
                    })
                results.append({
                    "changes": changes,
                    "new_input": new_input,
                    "n_flips": n_flips,
                })

        return results
