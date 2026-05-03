#!/usr/bin/env python
"""
SNA-STP: One-Click Experiment Reproduction
==========================================
Reproduces all experiments from the NeurIPS 2026 submission:
  "Algebraic Tomography for Exact Layer-Preserving Logic Extraction
   over Neural Bottleneck Circuits"

Usage:
  python run_all_experiments.py          # Run all experiments
  python run_all_experiments.py --quick  # Quick subset (~2 min)
  python run_all_experiments.py --e1     # Only Experiment 1
  python run_all_experiments.py --list   # List available experiments

Requirements: pip install -r requirements.txt
"""

import sys
import os
import time
import argparse
import traceback

sys.path.insert(0, os.path.dirname(__file__))

RESULTS = {}


def run_experiment(name, module_path, func_name=None):
    """Run a single experiment and record timing/status."""
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")
    t0 = time.time()
    try:
        import importlib
        mod = importlib.import_module(module_path.replace('/', '.').replace('.py', ''))
        if func_name:
            getattr(mod, func_name)()
        elif hasattr(mod, 'main'):
            mod.main()
        else:
            exec(open(os.path.join(os.path.dirname(__file__), module_path)).read())
        elapsed = time.time() - t0
        RESULTS[name] = f"PASSED ({elapsed:.1f}s)"
        print(f"\n  {name}: PASSED ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.time() - t0
        RESULTS[name] = f"FAILED ({e})"
        print(f"\n  {name}: FAILED ({elapsed:.1f}s)")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='SNA-STP Experiment Suite')
    parser.add_argument('--quick', action='store_true', help='Run quick subset')
    parser.add_argument('--e1', action='store_true', help='Only E1 (Exactness)')
    parser.add_argument('--e2', action='store_true', help='Only E2 (Baseline)')
    parser.add_argument('--e3', action='store_true', help='Only E3 (SAT Comparison)')
    parser.add_argument('--e4', action='store_true', help='Only E4 (Repair)')
    parser.add_argument('--e5', action='store_true', help='Only E5 (MNIST)')
    parser.add_argument('--e6', action='store_true', help='Only E6 (CART Baseline)')
    parser.add_argument('--e7', action='store_true', help='Only E7 (CIFAR-10 Bottleneck, GPU required)')
    parser.add_argument('--list', action='store_true', help='List experiments')
    args = parser.parse_args()

    experiments = [
        ("E1: Exactness & Attribution Collapse",
         "experiments/e1_exactness.py", "run_exactness_benchmark"),
        ("E2: Baseline Comparison (SHAP/LIME/IG/DT)",
         "experiments/e2_baseline.py", "run_comparison_experiment"),
        ("E3: SAT/SMT Head-to-Head Comparison",
         "experiments/e3_sat_comparison.py", "run_sat_comparison"),
        ("E4: Layer-wise Logic Repair",
         "experiments/e4_repair.py", "run_repair_experiment"),
        ("E5: MNIST & Synthetic Experiments",
         "experiments/e5_mnist.py", "run_mnist_experiments"),
        ("E6: CART Decision Tree Baseline",
         "experiments/e6_cart_baseline.py", None),
        ("E7: CIFAR-10 ResNet Bottleneck (GPU required, ~4-5h)",
         "experiments/e7_cifar_bottleneck.py", None),
    ]

    if args.list:
        for i, (name, path, func) in enumerate(experiments):
            print(f"  E{i+1}: {name}")
        return

    if args.quick:
        run_experiment("E1: Exactness", "experiments/e1_exactness.py", "run_exactness_benchmark")
        run_experiment("E4: Repair", "experiments/e4_repair.py", "run_repair_experiment")
    elif args.e1:
        run_experiment(*experiments[0])
    elif args.e2:
        run_experiment(*experiments[1])
    elif args.e3:
        run_experiment(*experiments[2])
    elif args.e4:
        run_experiment(*experiments[3])
    elif args.e5:
        run_experiment(*experiments[4])
    elif args.e6:
        run_experiment(*experiments[5])
    elif args.e7:
        run_experiment(*experiments[6])
    else:
        for name, path, func in experiments:
            if 'GPU required' in name:
                print(f"\n  Skipping {name} (requires GPU, use --e7 to run)")
                continue
            run_experiment(name, path, func)

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    for name, status in RESULTS.items():
        print(f"  {name}: {status}")


if __name__ == '__main__':
    main()
