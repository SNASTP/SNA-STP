"""
Feature Space Analyzer
======================

Performs STP analysis in the learned feature space rather than
the raw input space, enabling analysis of deep networks.

Strategy:
1. Use network's early layers as feature extractor (e.g., CNN convolutions)
2. Analyze the classifier head in the low-dimensional feature space
3. This reveals "what decision rules the network learned" without
   needing to enumerate all 2^n raw inputs

Best for: CNN + FC classifier, Transformer + classification head.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from ..core.stp import STPCore
from ..core.structure_matrix import StructureMatrixBuilder
from ..core.logic_extractor import LogicExtractor


class FeatureSpaceAnalyzer:
    """
    STP analysis in the trained feature space.

    For networks like CNN→FC or Transformer→FC, the pre-classifier
    features often have low dimensionality (8–64), making exact
    STP analysis feasible even when raw input dimension is huge.

    Examples:
        >>> # After splitting model into feature_extractor and classifier
        >>> analyzer = FeatureSpaceAnalyzer(
        ...     feature_extractor=cnn_backbone,
        ...     classifier=fc_head,
        ...     feature_dim=16
        ... )
        >>> result = analyzer.analyze(num_classes=10)
    """

    def __init__(
        self,
        feature_extractor=None,
        classifier=None,
        feature_dim: int = 16,
        threshold: float = 0.5,
        activation: str = "sigmoid",
    ):
        """
        Args:
            feature_extractor: PyTorch module that maps raw input → features.
            classifier: PyTorch module that maps features → output.
            feature_dim: Dimension of the feature space.
            threshold: Binarization threshold for features.
            activation: Activation function for the classifier.
        """
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.feature_dim = feature_dim
        self.threshold = threshold
        self.activation = activation

        self.stp = STPCore()
        self.builder = StructureMatrixBuilder(threshold=threshold, activation=activation)
        self.extractor = LogicExtractor(var_prefix="f")

    def analyze(
        self,
        num_classes: int = 2,
        feature_names: Optional[List[str]] = None,
        compute_shapley: bool = False,
    ) -> Dict:
        """
        Analyze the classifier's logic in feature space.

        Args:
            num_classes: Number of output classes.
            feature_names: Names for feature dimensions.
            compute_shapley: Whether to compute Shapley values.

        Returns:
            Dict with structure_matrix, rules, patterns, etc.
        """
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(self.feature_dim)]

        if self.feature_dim > 20:
            return self._sampled_analysis(num_classes, feature_names)
        else:
            return self._exact_analysis(num_classes, feature_names, compute_shapley)

    def _exact_analysis(
        self,
        num_classes: int,
        feature_names: List[str],
        compute_shapley: bool,
    ) -> Dict:
        """Full enumeration analysis in feature space."""
        import torch.nn as nn

        # Extract classifier weights
        classifier_params = list(self.classifier.parameters())
        if len(classifier_params) < 2:
            raise ValueError("Classifier needs weight and bias parameters")

        W = classifier_params[0].detach().cpu().numpy()
        b = classifier_params[1].detach().cpu().numpy()

        n_states = 2**self.feature_dim
        class_patterns = defaultdict(list)

        # Build per-class structure matrix
        if num_classes == 2 and W.shape[0] == 1:
            # Binary classification with single output
            M = np.zeros((2, n_states), dtype=np.float64)
            for j in range(n_states):
                bits = self._index_to_bits(j, self.feature_dim)
                fv = np.array(bits, dtype=np.float64)
                z = float(W[0] @ fv + b[0])
                a = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
                pred = 1 if a >= self.threshold else 0
                M[:, j] = STPCore.bool_to_delta(pred).flatten()
                class_patterns[pred].append(bits)

            rules = self.extractor.extract_rules(M, var_names=feature_names)
            shapley = None
            if compute_shapley and self.feature_dim <= 12:
                shapley = self.extractor.compute_shapley_from_matrix(
                    M, var_names=feature_names
                )

            return {
                "structure_matrix": M,
                "class_patterns": dict(class_patterns),
                "rules": rules,
                "shapley_values": shapley,
                "feature_dim": self.feature_dim,
                "mode": "exact",
            }
        else:
            # Multi-class
            results_per_class = {}
            for j in range(n_states):
                bits = self._index_to_bits(j, self.feature_dim)
                fv = np.array(bits, dtype=np.float64)
                z = W @ fv + b
                pred = int(np.argmax(z))
                class_patterns[pred].append(bits)

            for cls in range(num_classes):
                patterns = class_patterns.get(cls, [])
                results_per_class[cls] = {
                    "count": len(patterns),
                    "ratio": len(patterns) / n_states,
                    "sample_patterns": patterns[:5],
                }

            return {
                "class_patterns": dict(class_patterns),
                "class_summary": results_per_class,
                "feature_dim": self.feature_dim,
                "total_states": n_states,
                "mode": "exact",
                "num_classes": num_classes,
            }

    def _sampled_analysis(
        self, num_classes: int, feature_names: List[str], n_samples: int = 10000
    ) -> Dict:
        """Sampling-based analysis for high-dimensional feature spaces."""
        classifier_params = list(self.classifier.parameters())
        W = classifier_params[0].detach().cpu().numpy()
        b = classifier_params[1].detach().cpu().numpy()

        samples = np.random.randint(0, 2, size=(n_samples, self.feature_dim))
        class_counts = defaultdict(int)

        for sample in samples:
            z = W @ sample.astype(np.float64) + b
            if num_classes == 2 and W.shape[0] == 1:
                a = 1.0 / (1.0 + np.exp(-np.clip(z[0], -500, 500)))
                pred = 1 if a >= self.threshold else 0
            else:
                pred = int(np.argmax(z))
            class_counts[pred] += 1

        # Feature importance via weight magnitude
        if W.shape[0] == 1:
            importance = np.abs(W[0])
        else:
            importance = np.abs(W).mean(axis=0)
        top_features = np.argsort(importance)[::-1][:5]

        return {
            "sampled": True,
            "n_samples": n_samples,
            "class_distribution": dict(class_counts),
            "feature_importance": {
                feature_names[i]: float(importance[i])
                for i in top_features
            },
            "top_features": [feature_names[i] for i in top_features],
            "feature_dim": self.feature_dim,
            "mode": "sample",
        }

    def analyze_with_data(
        self,
        data_loader,
        num_classes: int = 2,
        max_samples: int = 1000,
    ) -> Dict:
        """
        Analyze using actual data through the feature extractor.

        Instead of enumerating binary feature vectors, this method
        passes real data through the feature extractor and analyzes
        the distribution of features and predictions.

        Args:
            data_loader: PyTorch DataLoader yielding (inputs, labels).
            num_classes: Number of classes.
            max_samples: Maximum samples to process.

        Returns:
            Dict with feature statistics and consistency analysis.
        """
        import torch

        if self.feature_extractor is None or self.classifier is None:
            raise ValueError("Both feature_extractor and classifier are required")

        self.feature_extractor.eval()
        self.classifier.eval()

        all_features = []
        all_labels = []
        all_nn_preds = []
        all_stp_preds = []

        classifier_params = list(self.classifier.parameters())
        W = classifier_params[0].detach().cpu().numpy()
        b = classifier_params[1].detach().cpu().numpy()

        n_processed = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                features = self.feature_extractor(inputs)
                logits = self.classifier(features)

                for i in range(len(inputs)):
                    if n_processed >= max_samples:
                        break

                    feat_continuous = features[i].cpu().numpy()
                    feat_binary = (feat_continuous > self.threshold).astype(int)

                    nn_pred = int(logits[i].argmax())
                    stp_logits = W @ feat_binary.astype(np.float64) + b
                    stp_pred = int(np.argmax(stp_logits))

                    all_features.append(feat_continuous)
                    all_labels.append(int(labels[i]))
                    all_nn_preds.append(nn_pred)
                    all_stp_preds.append(stp_pred)
                    n_processed += 1

                if n_processed >= max_samples:
                    break

        all_features = np.array(all_features)
        consistency = np.mean(np.array(all_nn_preds) == np.array(all_stp_preds))

        # Feature distribution analysis
        near_threshold = np.mean(
            np.any(np.abs(all_features - self.threshold) < 0.1, axis=1)
        )

        return {
            "n_samples": n_processed,
            "consistency": float(consistency),
            "near_threshold_ratio": float(near_threshold),
            "feature_mean": all_features.mean(axis=0).tolist(),
            "feature_std": all_features.std(axis=0).tolist(),
            "nn_accuracy": float(
                np.mean(np.array(all_nn_preds) == np.array(all_labels))
            ),
        }

    def _index_to_bits(self, j: int, n: int) -> List[int]:
        bits = []
        for i in range(n):
            bit = 1 - ((j >> (n - 1 - i)) & 1)
            bits.append(bit)
        return bits
