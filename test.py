"""
Assignment #3 — Problem 2: Marabou Verification on External Model

Verifies local robustness of a trained MNIST MLP using Marabou.

Robustness property:
  For a test image x classified as digit d, all perturbed inputs x' satisfying
  ||x' - x||_inf <= eps must also be classified as digit d.

This is encoded as a Marabou satisfiability query:
  - Input bounds: x_i - eps <= x'_i <= x_i + eps  (clipped to [0, 1])
  - Output constraint: for a chosen competing class j, output[j] >= output[d]
  If the query is UNSAT, no such perturbation exists (robust for that class).
  If SAT, Marabou returns an adversarial example.

Running all 9 sub-queries (one per competing class) gives a complete robustness
certificate. Use --full to enable this; the default checks only the top competitor.

Usage:
    python test.py
    python test.py --index 7 --eps 0.005 --full
"""

import os
import sys
import time
import argparse
import numpy as np
import torchvision
import torchvision.transforms as transforms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_test_sample(index: int):
    """Load one MNIST test image; return (flat float64 array of shape (784,), int label)."""
    transform = transforms.Compose([transforms.ToTensor()])
    test_ds = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    image, label = test_ds[index]
    x = image.numpy().flatten().astype(np.float64)
    return x, int(label)


def ort_predict(x: np.ndarray, model_path: str) -> tuple[int, np.ndarray]:
    """Run inference with ONNX Runtime; return (predicted class, logits array)."""
    import onnxruntime as ort
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    x_in = x.reshape(1, 784).astype(np.float32)
    logits = sess.run(None, {"input": x_in})[0][0]   # shape (10,)
    return int(np.argmax(logits)), logits


# ---------------------------------------------------------------------------
# Single sub-query: can class j beat true_label within the eps-ball?
# ---------------------------------------------------------------------------

def check_one_class(
    x: np.ndarray,
    true_label: int,
    challenger: int,
    eps: float,
    model_path: str,
    timeout: int = 300,
) -> tuple[str, dict | None]:
    """
    Ask Marabou: does an adversarial x' exist where output[challenger] >= output[true_label]?

    Returns:
        ('sat',  variable_assignment_dict)  — adversarial example found
        ('unsat', None)                     — no such perturbation exists
        ('timeout', None)                   — query exceeded timeout
        ('error', None)                     — unexpected result code
    """
    from maraboupy import Marabou, MarabouUtils

    # Load the ONNX model freshly for each sub-query
    network = Marabou.read_onnx(model_path)

    # Flatten variable arrays (shape depends on Marabou version)
    input_vars = network.inputVars[0].flatten()
    output_vars = network.outputVars[0].flatten()

    # --- Input bounds: L-inf ball clipped to valid pixel range [0, 1] ---
    for i, var in enumerate(input_vars):
        lb = float(max(0.0, x[i] - eps))
        ub = float(min(1.0, x[i] + eps))
        network.setLowerBound(var, lb)
        network.setUpperBound(var, ub)

    # --- Output constraint: output[challenger] - output[true_label] >= 0 ---
    # This asks: can the challenger class beat the true class?
    eq = MarabouUtils.Equation(MarabouUtils.Equation.EquationType.GE)
    eq.addAddend(1,  int(output_vars[challenger]))   # +1 * output[challenger]
    eq.addAddend(-1, int(output_vars[true_label]))   # -1 * output[true_label]
    eq.setScalar(0.0)                                # >= 0
    network.addEquation(eq)

    options = Marabou.createOptions(verbosity=0, timeoutInSeconds=timeout)
    exit_code, vals, stats = network.solve(options=options)

    exit_code = exit_code.lower().strip()
    if exit_code == "sat":
        return "sat", vals
    elif exit_code == "unsat":
        return "unsat", None
    elif "timeout" in exit_code:
        return "timeout", None
    else:
        return "error", None


# ---------------------------------------------------------------------------
# Robustness verification entry point
# ---------------------------------------------------------------------------

def verify_robustness(
    x: np.ndarray,
    true_label: int,
    logits: np.ndarray,
    eps: float,
    model_path: str,
    full: bool = False,
    timeout: int = 300,
) -> dict:
    """
    Verify local robustness.

    If full=False, only the top competitor (second-highest logit class) is checked.
    If full=True,  all 9 competing classes are checked.

    Returns a result dict with keys: status, elapsed, adversarial_class, adversarial_input.
    """
    n_classes = len(logits)

    # Rank competing classes by logit (highest = hardest adversary, check first)
    ranking = sorted(
        [j for j in range(n_classes) if j != true_label],
        key=lambda j: -logits[j],
    )
    classes_to_check = ranking if full else [ranking[0]]

    start = time.time()
    for j in classes_to_check:
        print(f"  Checking class {j} (logit={logits[j]:.3f}) vs true class {true_label} ...")
        result, vals = check_one_class(x, true_label, j, eps, model_path, timeout)

        if result == "sat":
            # Build adversarial input from Marabou's variable assignment
            network_tmp = __import__("maraboupy").Marabou.read_onnx(model_path)
            input_vars = network_tmp.inputVars[0].flatten()
            adv = np.array([vals.get(int(v), 0.0) for v in input_vars], dtype=np.float64)
            return {
                "status": "SAT",
                "elapsed": time.time() - start,
                "adversarial_class": j,
                "adversarial_input": adv,
            }
        elif result == "timeout":
            print(f"    [TIMEOUT] exceeded {timeout}s for class {j}")
        elif result == "error":
            print(f"    [ERROR] unexpected exit code for class {j}")

    return {
        "status": "UNSAT",
        "elapsed": time.time() - start,
        "adversarial_class": None,
        "adversarial_input": None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Marabou robustness verification for MNIST MLP"
    )
    parser.add_argument(
        "--model", type=str, default="models/mnist_mlp.onnx",
        help="Path to the ONNX model (default: models/mnist_mlp.onnx)",
    )
    parser.add_argument(
        "--index", type=int, default=0,
        help="MNIST test sample index (default: 0)",
    )
    parser.add_argument(
        "--eps", type=float, default=0.01,
        help="L-inf perturbation radius (default: 0.01)",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Check all 9 competing classes instead of only the top competitor",
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Per-query timeout in seconds (default: 300)",
    )
    args = parser.parse_args()

    # Auto-train if model is missing
    if not os.path.exists(args.model):
        print(f"[INFO] Model not found at '{args.model}'. Running train_model.py first...\n")
        import subprocess
        subprocess.run([sys.executable, "train_model.py"], check=True)

    print("=" * 60)
    print("  Marabou MNIST Robustness Verification")
    print("=" * 60)
    print(f"  Model   : {args.model}")
    print(f"  Sample  : index={args.index}")
    print(f"  Epsilon : {args.eps}")
    print(f"  Mode    : {'full (all 9 classes)' if args.full else 'fast (top competitor only)'}")
    print(f"  Timeout : {args.timeout}s per query")
    print()

    # Load sample
    x, true_label = get_test_sample(args.index)
    predicted, logits = ort_predict(x, args.model)

    print(f"  True label      : {true_label}")
    print(f"  Predicted label : {predicted}")

    if predicted != true_label:
        print("\n[WARNING] The model misclassifies this sample.")
        print("  Robustness is only meaningful for correctly classified inputs.")
        print("  Consider using a different --index.\n")

    print()
    print("Running Marabou verification...")
    print()

    result = verify_robustness(
        x, true_label, logits, args.eps, args.model,
        full=args.full, timeout=args.timeout,
    )

    # Report
    print()
    print("=" * 60)
    print("  Result")
    print("=" * 60)
    print(f"  Outcome : {result['status']}")
    print(f"  Time    : {result['elapsed']:.2f}s")

    if result["status"] == "UNSAT":
        mode_desc = "all classes" if args.full else "top competitor class"
        print(f"\n  [VERIFIED] All inputs within eps={args.eps} of x (for {mode_desc})")
        print(f"  are classified as digit {true_label}.")
    elif result["status"] == "SAT":
        j = result["adversarial_class"]
        adv = result["adversarial_input"]
        linf = float(np.max(np.abs(adv - x)))
        print(f"\n  [VIOLATED] Adversarial example found — model predicts class {j}.")
        print(f"  Max pixel perturbation: {linf:.6f}  (eps={args.eps})")
    print()


if __name__ == "__main__":
    main()
