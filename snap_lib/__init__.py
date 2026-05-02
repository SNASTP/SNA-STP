"""
SNA-STP: Structural Neural Analysis via Semi-tensor Product
=========================================================

A Python library for neural network interpretability through
Semi-Tensor Product (STP) based structural logic analysis.

Core Capabilities:
- Extract boolean logic rules from trained neural networks
- Compute structure matrices for complete input-output mapping
- Provide exact Shapley values from structure matrices
- Support MLP, CNN, RNN, and Attention architectures
- Multiple analysis modes: Exact, Sampled, Local

Usage:
    >>> from snap_lib import SNAPAnalyzer
    >>> analyzer = SNAPAnalyzer(model)
    >>> result = analyzer.analyze()
    >>> print(result.logic_rules)

Author: SNA-STP Research Team
License: MIT
"""

__version__ = "0.1.0"

from snap_lib.core.stp import STPCore
from snap_lib.core.structure_matrix import StructureMatrixBuilder
from snap_lib.core.logic_extractor import LogicExtractor
from snap_lib.analyzers.snap_analyzer import SNAPAnalyzer
from snap_lib.analyzers.layerwise import LayerwiseAnalyzer
from snap_lib.analyzers.feature_space import FeatureSpaceAnalyzer

__all__ = [
    "STPCore",
    "StructureMatrixBuilder",
    "LogicExtractor",
    "SNAPAnalyzer",
    "LayerwiseAnalyzer",
    "FeatureSpaceAnalyzer",
]
