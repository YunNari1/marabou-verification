"""
Problem 1: Exploration of the Marabou resources directory.

Summarizes the models, datasets, and example verification queries
available in the official Marabou GitHub repository.

Usage:
    python explore_resources.py
    python explore_resources.py --marabou-dir /path/to/Marabou
"""

import os
import argparse

# ---------------------------------------------------------------------------
# Known contents of the Marabou resources/ directory (from GitHub inspection)
# Source: https://github.com/NeuralNetworkVerification/Marabou/tree/master/resources
# ---------------------------------------------------------------------------

KNOWN_MODELS = {
    "NNet (.nnet)": {
        "description": (
            "Custom text-based format for fully-connected ReLU networks. "
            "Encodes weights, biases, and input normalization."
        ),
        "examples": [
            "ACAS Xu networks: ACASXU_experimental_v2a_X_Y.nnet (X=1..5, Y=1..9)",
            "Total 45 networks — aircraft collision avoidance advisory system",
            "Architecture: 5 inputs → 5 hidden layers × 50 neurons (ReLU) → 5 outputs",
            "Input: (rho, theta, psi, v_own, v_int) — relative aircraft state",
            "Output: 5 advisory actions (e.g., Clear-of-Conflict, weak/strong left/right)",
        ],
    },
    "ONNX (.onnx)": {
        "description": (
            "Open Neural Network Exchange format. Supports diverse architectures "
            "including fully-connected networks and small CNNs."
        ),
        "examples": [
            "Small MNIST classifiers (fully connected, ~784→X→10)",
            "Simple test networks used for correctness checks",
            "Networks from VNN-COMP (Verification of Neural Networks Competition)",
        ],
    },
    "TensorFlow / Keras (.pb, SavedModel)": {
        "description": (
            "TensorFlow protocol buffer or SavedModel format. "
            "Support is more limited compared to NNet and ONNX."
        ),
        "examples": [
            "Small TF models for testing the maraboupy TF reader",
        ],
    },
}

KNOWN_DATASETS = [
    "ACAS Xu input space: 5 continuous features representing aircraft geometry",
    "MNIST: 28×28 grayscale digit images — accessed externally via torchvision/tf.keras",
    "Custom input specifications embedded in .txt property files (lower/upper bounds per feature)",
]

KNOWN_QUERIES = [
    {
        "name": "ACAS Xu Safety Properties (phi1 – phi10)",
        "description": (
            "Ten formally specified safety properties from Katz et al. (CAV 2019). "
            "Each defines an input region (a hyperrectangle in the 5D state space) "
            "and an output constraint (e.g., a particular action must not be taken, "
            "or must be the minimum-score output)."
        ),
        "query_type": "Safety / reachability",
        "constraint_format": "Input box constraints + output index ordering or threshold",
        "result_interpretation": (
            "UNSAT → property holds for all inputs in the region; "
            "SAT → a violating input (counterexample) was found"
        ),
    },
    {
        "name": "Local Robustness (L-inf ball)",
        "description": (
            "Given an input x and a perturbation radius ε, verify that all x' "
            "satisfying ‖x' − x‖∞ ≤ ε are classified identically to x. "
            "Encoded by bounding each input variable to [x_i − ε, x_i + ε] and "
            "adding output constraints that rule out any competing class winning."
        ),
        "query_type": "Robustness",
        "constraint_format": "Per-feature interval bounds + output difference inequalities",
        "result_interpretation": (
            "UNSAT → model is locally robust at x; "
            "SAT → an adversarial perturbation within ε exists"
        ),
    },
    {
        "name": "Reachability / Output Range",
        "description": (
            "Determine whether a given output value (or range of output values) "
            "is reachable from some input in a specified region."
        ),
        "query_type": "Reachability",
        "constraint_format": "Input region bounds + output bound constraint",
        "result_interpretation": (
            "SAT → the output range is reachable; "
            "UNSAT → the output is provably outside the queried range"
        ),
    },
]


def explore_local(marabou_dir: str):
    """Walk the local Marabou resources directory and print the file tree."""
    resources_path = os.path.join(marabou_dir, "resources")
    if not os.path.exists(resources_path):
        print(f"[WARNING] resources/ not found at: {resources_path}")
        return

    print(f"\n{'='*60}")
    print(f"  Local Marabou Resources Tree")
    print(f"  Path: {resources_path}")
    print(f"{'='*60}\n")

    ext_counts: dict[str, int] = {}
    for root, dirs, files in os.walk(resources_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        level = root.replace(resources_path, '').count(os.sep)
        indent = "  " * level
        rel = os.path.relpath(root, resources_path)
        label = rel if rel != '.' else 'resources/'
        print(f"{indent}[{label}]")
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower() or '(no ext)'
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
            print(f"{indent}  {fname}")

    print("\n  File counts by extension:")
    for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1]):
        print(f"    {ext:10s} {count} file(s)")


def print_summary():
    """Print the documented summary of Marabou resources."""
    print("=" * 65)
    print("  Marabou Resources Directory — Summary (Problem 1)")
    print("  Repo: https://github.com/NeuralNetworkVerification/Marabou")
    print("=" * 65)

    # Models
    print("\n[1] Model Types & Formats")
    print("-" * 40)
    for fmt, info in KNOWN_MODELS.items():
        print(f"\n  Format : {fmt}")
        print(f"  Info   : {info['description']}")
        print(f"  Examples:")
        for ex in info["examples"]:
            print(f"    • {ex}")

    # Datasets
    print("\n[2] Datasets / Input Specifications")
    print("-" * 40)
    for d in KNOWN_DATASETS:
        print(f"  • {d}")

    # Verification queries
    print("\n[3] Example Verification Queries")
    print("-" * 40)
    for i, q in enumerate(KNOWN_QUERIES, 1):
        print(f"\n  ({i}) {q['name']}")
        print(f"      Type   : {q['query_type']}")
        print(f"      Desc   : {q['description']}")
        print(f"      Format : {q['constraint_format']}")
        print(f"      Result : {q['result_interpretation']}")

    print("\n[4] Summary Table")
    print("-" * 40)
    print(f"  {'Format':<12} {'Networks':<10} {'Primary Use Case'}")
    print(f"  {'-'*12} {'-'*10} {'-'*30}")
    print(f"  {'NNet':<12} {'45':<10} ACAS Xu safety verification")
    print(f"  {'ONNX':<12} {'several':<10} MNIST, VNN-COMP benchmarks")
    print(f"  {'TF/Keras':<12} {'a few':<10} TF model loading tests")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize Marabou resources directory (Assignment 3, Problem 1)"
    )
    parser.add_argument(
        "--marabou-dir",
        type=str,
        default=None,
        help="Path to a local Marabou clone to explore its resources/ directory",
    )
    args = parser.parse_args()

    print_summary()

    if args.marabou_dir:
        explore_local(args.marabou_dir)
    else:
        print(
            "\n[Tip] Pass --marabou-dir /path/to/Marabou to also list local resource files."
        )


if __name__ == "__main__":
    main()
