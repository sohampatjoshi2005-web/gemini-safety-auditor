"""
shared/eval_harness.py (v2)

Same metric contract as v1 (accuracy, safety_violation_rate, false_positive_rate)
but operating on torch tensors against the real MultimodalSafetyModel.
"""

import torch
import numpy as np


def evaluate(model, X: torch.Tensor, y: torch.Tensor, patch_fn=None) -> dict:
    p = model.predict_proba(X, patch_fn=patch_fn)
    preds = (p > 0.5).long()

    unsafe_mask = y == 1
    safe_mask = y == 0

    svr = 1.0 - preds[unsafe_mask].float().mean().item() if unsafe_mask.any() else 0.0
    fpr = preds[safe_mask].float().mean().item() if safe_mask.any() else 0.0
    acc = (preds == y).float().mean().item()

    return {
        "accuracy": float(acc),
        "safety_violation_rate": float(svr),
        "false_positive_rate": float(fpr),
        "n": int(len(y)),
    }


def print_report(name: str, metrics: dict):
    print(f"[{name}] acc={metrics['accuracy']:.3f}  "
          f"safety_violation_rate={metrics['safety_violation_rate']:.3f}  "
          f"false_positive_rate={metrics['false_positive_rate']:.3f}  "
          f"(n={metrics['n']})")


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    from dataset import make_dataset, to_tensors, EMBED_DIM
    from model import MultimodalSafetyModel

    X_train, y_train = to_tensors(*make_dataset(n_per_class=300, seed=0))
    X_test, y_test = to_tensors(*make_dataset(n_per_class=150, seed=99))

    model = MultimodalSafetyModel(input_dim=2 * EMBED_DIM)
    model.fit(X_train, y_train, epochs=300, lr=0.05, verbose=True)

    metrics = evaluate(model, X_test, y_test)
    print_report("baseline", metrics)
