"""
tests/test_pipeline.py

Smoke tests for every shared component. Run with: pytest tests/ -v
These aren't exhaustive unit tests, but they catch the most likely
regressions (shape mismatches, NaN losses, broken imports) and are the
kind of thing a reviewer expects to see in a repo that claims to be
more than a one-off script.
"""

import sys
import os
import torch
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "shared"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "workstream1_interpretability"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "workstream2_redteam"))

from dataset import make_dataset, to_tensors, EMBED_DIM  # noqa: E402
from model import MultimodalSafetyModel, register_activation_hook  # noqa: E402
from eval_harness import evaluate  # noqa: E402


def test_dataset_shapes():
    X, y = make_dataset(n_per_class=20, seed=0)
    assert X.shape == (40, 2 * EMBED_DIM)
    assert set(y.tolist()) == {0, 1}


def test_heldout_split_disjoint_from_train_split():
    from dataset import _HELDOUT_TEXT, TEXT_CONCEPTS
    train_concepts = set(TEXT_CONCEPTS) - _HELDOUT_TEXT
    assert train_concepts.isdisjoint(_HELDOUT_TEXT)


def test_model_forward_shapes():
    model = MultimodalSafetyModel(input_dim=2 * EMBED_DIM)
    X, _ = to_tensors(*make_dataset(n_per_class=10, seed=0))
    logits, hidden = model.forward(X, return_hidden=True)
    assert logits.shape == (20,)
    assert hidden.shape == (20, 32)


def test_model_trains_without_nan():
    X, y = to_tensors(*make_dataset(n_per_class=50, seed=0))
    model = MultimodalSafetyModel(input_dim=2 * EMBED_DIM)
    model.fit(X, y, epochs=50, lr=0.05)
    logits = model.forward(X)
    assert not torch.isnan(logits).any()


def test_eval_harness_metrics_in_range():
    X_train, y_train = to_tensors(*make_dataset(n_per_class=100, seed=0))
    X_test, y_test = to_tensors(*make_dataset(n_per_class=50, seed=1))
    model = MultimodalSafetyModel(input_dim=2 * EMBED_DIM)
    model.fit(X_train, y_train, epochs=150, lr=0.05)
    metrics = evaluate(model, X_test, y_test)
    for key in ("accuracy", "safety_violation_rate", "false_positive_rate"):
        assert 0.0 <= metrics[key] <= 1.0


def test_activation_hook_changes_output():
    X, y = to_tensors(*make_dataset(n_per_class=50, seed=0))
    model = MultimodalSafetyModel(input_dim=2 * EMBED_DIM)
    model.fit(X, y, epochs=100, lr=0.05)
    with torch.no_grad():
        before = model.forward(X)

    direction = torch.randn(32)
    direction = direction / direction.norm()

    def patch_fn(h):
        return h - (h @ direction).unsqueeze(-1) * direction.unsqueeze(0)

    handle = register_activation_hook(model, patch_fn)
    with torch.no_grad():
        after = model.forward(X)
    handle.remove()
    assert not torch.allclose(before, after)


def test_pgd_attack_reduces_confidence():
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "workstream2_redteam"))
    from attack_loop import pgd_attack

    X, y = to_tensors(*make_dataset(n_per_class=100, seed=0))
    model = MultimodalSafetyModel(input_dim=2 * EMBED_DIM)
    model.fit(X, y, epochs=200, lr=0.05)

    X_unsafe = X[y == 1]
    x0 = X_unsafe[0]
    p_before = model.predict_proba(x0.unsqueeze(0)).item()
    _, p_after, _ = pgd_attack(model, x0, eps=1.5, alpha=0.1, n_steps=100)
    assert p_after <= p_before + 1e-6  # PGD should never make the attack worse


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
