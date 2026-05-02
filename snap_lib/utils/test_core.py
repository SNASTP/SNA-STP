"""
Unit tests for snap_lib core components.
Tests STP operations, structure matrix building, Shapley computation,
and the new per-instance Shapley method.
"""

import numpy as np
import torch
import torch.nn as nn
import unittest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from snap_lib.core.stp import STPCore
from snap_lib.core.structure_matrix import StructureMatrixBuilder
from snap_lib.core.logic_extractor import LogicExtractor


class TestSTPCore(unittest.TestCase):
    """Test fundamental STP operations."""

    def setUp(self):
        self.stp = STPCore()

    def test_bool_to_delta(self):
        d1 = self.stp.bool_to_delta(1)
        d0 = self.stp.bool_to_delta(0)
        np.testing.assert_array_equal(d1.flatten(), [1, 0])
        np.testing.assert_array_equal(d0.flatten(), [0, 1])

    def test_delta_to_bool(self):
        self.assertEqual(self.stp.delta_to_bool(np.array([[1], [0]])), 1)
        self.assertEqual(self.stp.delta_to_bool(np.array([[0], [1]])), 0)

    def test_encode_state_single(self):
        s1 = self.stp.encode_state([1])
        np.testing.assert_array_equal(s1.flatten(), [1, 0])
        s0 = self.stp.encode_state([0])
        np.testing.assert_array_equal(s0.flatten(), [0, 1])

    def test_encode_state_multi(self):
        # [1, 1] → δ₄¹ = [1,0,0,0]
        s = self.stp.encode_state([1, 1])
        np.testing.assert_array_equal(s.flatten(), [1, 0, 0, 0])
        # [1, 0] → δ₄² = [0,1,0,0]
        s = self.stp.encode_state([1, 0])
        np.testing.assert_array_equal(s.flatten(), [0, 1, 0, 0])
        # [0, 0] → δ₄⁴ = [0,0,0,1]
        s = self.stp.encode_state([0, 0])
        np.testing.assert_array_equal(s.flatten(), [0, 0, 0, 1])

    def test_and_gate(self):
        M_and = STPCore.LOGIC_GATES['AND']  # [[1,0,0,0],[0,1,1,1]]
        for a in [0, 1]:
            for b in [0, 1]:
                x = self.stp.encode_state([a, b])
                y = M_and @ x
                result = self.stp.delta_to_bool(y)
                self.assertEqual(result, a & b,
                                 f"AND({a},{b}) should be {a & b}, got {result}")

    def test_or_gate(self):
        M_or = STPCore.LOGIC_GATES['OR']
        for a in [0, 1]:
            for b in [0, 1]:
                x = self.stp.encode_state([a, b])
                y = M_or @ x
                result = self.stp.delta_to_bool(y)
                self.assertEqual(result, a | b)

    def test_xor_gate(self):
        M_xor = STPCore.LOGIC_GATES['XOR']
        for a in [0, 1]:
            for b in [0, 1]:
                x = self.stp.encode_state([a, b])
                y = M_xor @ x
                result = self.stp.delta_to_bool(y)
                self.assertEqual(result, a ^ b)

    def test_stp_multiply_dimension_match(self):
        """When dimensions match, STP = standard matrix multiply."""
        A = np.array([[1, 0, 0, 0], [0, 1, 1, 1]])
        x = self.stp.encode_state([1, 1])
        result = self.stp.multiply(A, x)
        expected = A @ x
        np.testing.assert_array_almost_equal(result, expected)

    def test_khatri_rao(self):
        """Khatri-Rao product of two 2x4 matrices."""
        M_and = STPCore.LOGIC_GATES['AND']
        M_or = STPCore.LOGIC_GATES['OR']
        KR = self.stp.khatri_rao_product(M_and, M_or)
        self.assertEqual(KR.shape, (4, 4))  # (2*2) x 4


class TestStructureMatrix(unittest.TestCase):
    """Test structure matrix construction from neural networks."""

    def _make_xor_model(self):
        """Create a model trained on XOR."""
        model = nn.Sequential(
            nn.Linear(2, 4),
            nn.Sigmoid(),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )
        X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        criterion = nn.BCELoss()
        for _ in range(3000):
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            if loss.item() < 0.01:
                break
        return model

    def test_xor_structure_matrix(self):
        model = self._make_xor_model()
        builder = StructureMatrixBuilder(threshold=0.5, activation='sigmoid')
        result = builder.build_from_pytorch(model, activation='sigmoid')
        M = result['global_matrix']
        self.assertIsNotNone(M)
        self.assertEqual(M.shape[0], 2)
        self.assertEqual(M.shape[1], 4)  # 2^2

    def test_xor_correctness(self):
        """The structure matrix should recover XOR truth table."""
        model = self._make_xor_model()
        builder = StructureMatrixBuilder(threshold=0.5, activation='sigmoid')
        result = builder.build_from_pytorch(model, activation='sigmoid')
        M = result['global_matrix']

        stp = STPCore()
        # Check all 4 inputs
        expected = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}
        for (a, b), exp_out in expected.items():
            x = stp.encode_state([a, b])
            y = M @ x
            out = stp.delta_to_bool(y)
            self.assertEqual(out, exp_out,
                             f"XOR({a},{b}): expected {exp_out}, got {out}")


class TestShapleyValues(unittest.TestCase):
    """Test Shapley value computation."""

    def test_shapley_symmetry(self):
        """For symmetric gate (AND), both features should have equal Shapley."""
        M_and = STPCore.LOGIC_GATES['AND']
        extractor = LogicExtractor()
        shapley = extractor.compute_shapley_from_matrix(M_and, var_names=['x1', 'x2'])
        # AND gate: x1 and x2 are symmetric
        self.assertAlmostEqual(shapley['x1'], shapley['x2'], places=6)

    def test_shapley_sum_property(self):
        """Shapley efficiency: sum of φᵢ = v(N) - v(∅)."""
        M_xor = STPCore.LOGIC_GATES['XOR']
        extractor = LogicExtractor()
        shapley = extractor.compute_shapley_from_matrix(M_xor, var_names=['x1', 'x2'])
        # v(N) = fraction of outputs=1 when all features set to 1 → XOR(1,1)=0
        # v(∅) = fraction of outputs=1 when marginalized → average over all 4 inputs = 0.5
        # Actually v(N) = avg when all features present and set to 1 = XOR(1,1)=0
        # v(∅) = avg over all inputs = (0+1+1+0)/4 = 0.5
        # Sum should be v(N) - v(∅) = 0 - 0.5 = -0.5
        total = sum(shapley.values())
        self.assertAlmostEqual(total, -0.5, places=6)

    def test_shapley_returns_dict(self):
        M_and = STPCore.LOGIC_GATES['AND']
        extractor = LogicExtractor()
        shapley = extractor.compute_shapley_from_matrix(M_and)
        self.assertIsInstance(shapley, dict)
        self.assertEqual(len(shapley), 2)

    def test_instance_shapley_returns_dict(self):
        M_and = STPCore.LOGIC_GATES['AND']
        extractor = LogicExtractor()
        shapley = extractor.compute_instance_shapley(M_and, [1, 1])
        self.assertIsInstance(shapley, dict)
        self.assertEqual(len(shapley), 2)

    def test_instance_shapley_different_for_different_inputs(self):
        """Per-instance Shapley should vary with the input."""
        M_and = STPCore.LOGIC_GATES['AND']
        extractor = LogicExtractor()
        s_11 = extractor.compute_instance_shapley(M_and, [1, 1], var_names=['x1', 'x2'])
        s_00 = extractor.compute_instance_shapley(M_and, [0, 0], var_names=['x1', 'x2'])
        # AND(1,1)=1, AND(0,0)=0 — Shapley values for different instances should differ
        self.assertNotAlmostEqual(s_11['x1'], s_00['x1'], places=6)

    def test_instance_shapley_efficiency(self):
        """Per-instance Shapley efficiency: sum φᵢ(x) = f(x) - E[f]."""
        M_xor = STPCore.LOGIC_GATES['XOR']
        extractor = LogicExtractor()
        stp = STPCore()

        for bits in [[0, 0], [0, 1], [1, 0], [1, 1]]:
            shapley = extractor.compute_instance_shapley(M_xor, bits, var_names=['x1', 'x2'])
            total = sum(shapley.values())

            # f(x) from structure matrix
            x = stp.encode_state(bits)
            y = M_xor @ x
            f_x = float(y[0])

            # E[f] = average over all inputs
            e_f = 0.0
            for j in range(4):
                b = [1 - ((j >> (1 - k)) & 1) for k in range(2)]
                xs = stp.encode_state(b)
                ys = M_xor @ xs
                e_f += float(ys[0])
            e_f /= 4.0

            self.assertAlmostEqual(total, f_x - e_f, places=6,
                                   msg=f"Efficiency violated for input {bits}")


class TestLogicExtraction(unittest.TestCase):
    """Test DNF/CNF logic extraction."""

    def test_and_dnf(self):
        M_and = STPCore.LOGIC_GATES['AND']
        extractor = LogicExtractor()
        result = extractor.extract_rules(M_and, var_names=['x1', 'x2'])
        self.assertIn('dnf', result)
        # AND should have exactly one true pattern: [1, 1]
        self.assertEqual(len(result['true_patterns']), 1)

    def test_or_dnf(self):
        M_or = STPCore.LOGIC_GATES['OR']
        extractor = LogicExtractor()
        result = extractor.extract_rules(M_or, var_names=['x1', 'x2'])
        # OR should have 3 true patterns
        self.assertEqual(len(result['true_patterns']), 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
