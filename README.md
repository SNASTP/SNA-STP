# SNA-STP: Structural Neural Analysis via Semi-Tensor Product

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reference implementation for the NeurIPS 2026 submission:

> **Algebraic Tomography for Exact Layer-Preserving Logic Extraction over Neural Bottleneck Circuits: Structural Neural Analysis via Semi-Tensor Product**

SNA-STP is an algebraic tomography framework that extracts exact, layer-preserving Boolean logic from quantized neural bottleneck circuits using the semi-tensor product (STP). It achieves zero-error global logic extraction for binary neural networks and controlled consistency bounds for continuous quantized networks, while fully preserving internal layer topology.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all experiments
python run_all_experiments.py

# 3. Quick subset (~2 minutes)
python run_all_experiments.py --quick

# 4. List available experiments
python run_all_experiments.py --list
```

**Requirements:** Python 3.8+, numpy, torch, scipy, scikit-learn, shap, matplotlib, seaborn (see `requirements.txt`).

---

## Repository Structure

```
.
├── run_all_experiments.py    # One-click experiment runner
├── requirements.txt          # Python dependencies
├── config.py                 # Global configuration
├── snap_lib/                 # Core SNA-STP library
│   ├── core/
│   │   ├── stp.py            # Semi-tensor product algebra
│   │   ├── structure_matrix.py  # Boolean structure matrix construction
│   │   └── logic_extractor.py   # DNF/CNF logic rule extraction
│   └── analyzers/
│       ├── snap_analyzer.py     # High-level analysis API (Exact/Sample/Local)
│       ├── layerwise.py         # Per-layer structure matrix analysis
│       └── feature_space.py     # Feature-space analysis
├── experiments/              # Reproducible experiment scripts
│   ├── e1_exactness.py       # E1: Zero-error extraction & attribution collapse
│   ├── e2_baseline.py        # E2: Baseline comparison (SHAP/LIME/IG/DT)
│   ├── e3_sat_comparison.py  # E3: SAT/SMT head-to-head comparison + Z3 verification
│   ├── e4_repair.py          # E4: Layer-wise logic repair via matrix editing
│   ├── e5_mnist.py           # E5: MNIST & synthetic experiments
│   └── e6_cart_baseline.py   # E6: CART decision tree distillation baseline
└── data/                     # Datasets (auto-downloaded via sklearn/openml)
```

---

## Experiments

| # | Experiment | Paper Section | Key Result | Runtime |
|---|---|---|---|---|
| E1 | Exactness & Attribution Collapse | §4.2 | 100% BNN consistency vs SHAP/LIME collapse | ~5 min |
| E2 | Baseline Comparison | §4.2, App A | Full fidelity/sparsity/time comparison (6 methods) | ~15 min |
| E3 | SAT/SMT Head-to-Head | §4.7, App B.8 | SNA-STP-Exact vs Z3: property verification, DNF extraction, fault localization | ~3 min |
| E4 | Layer-wise Logic Repair | §4.6, App D.6 | Column-level surgical repair vs fine-tuning catastrophic forgetting | ~5 min |
| E5 | MNIST & Synthetic | §4.3, App A.3 | Multi-class MNIST, scalability, known-logic validation | ~10 min |
| E6 | CART Baseline | §4.2 | CART distillation fidelity (86.5% at interpretable depth) vs SNA-STP 100% | ~3 min |

---

## Citation

```bibtex
@inproceedings{snastp2026,
  title     = {Algebraic Tomography for Exact Layer-Preserving Logic Extraction
               over Neural Bottleneck Circuits: A Semi-Tensor-Product Approach},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
}
```

## License

MIT License. See `LICENSE` file for details.
