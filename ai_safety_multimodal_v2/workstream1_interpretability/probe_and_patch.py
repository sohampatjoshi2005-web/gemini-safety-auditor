"""
workstream1_interpretability/probe_and_patch.py (v2)

Upgrades over v1:
  - activation patching now goes through a real forward hook
    (register_activation_hook), the same mechanism you'd use on a real
    transformer layer, not just a function argument
  - the SAE is now a real nn.Module trained with Adam + a proper L1 sparsity
    penalty, and we report % of features that are "dead" (never fire) --
    a standard real-SAE health metric that v1 didn't track
  - NEW: we test the probe direction's generalization to held-out concepts
    (dataset.py's split="heldout_concepts") -- does the "unsafe direction"
    generalize to concepts the model never trained on, or is it memorizing
    the training vocabulary? This is a materially stronger interpretability
    claim than v1 made.
  - saves a plot (results/ws1_probe_projection.png) showing the projection
    of hidden activations onto the unsafe direction, safe vs unsafe, iid vs
    held-out concepts
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "shared"))
from dataset import make_dataset, to_tensors, EMBED_DIM  # noqa: E402
from model import MultimodalSafetyModel, register_activation_hook  # noqa: E402
from eval_harness import evaluate, print_report  # noqa: E402
from utils import save_result, save_fig  # noqa: E402


def train_probe(hidden: np.ndarray, labels: np.ndarray) -> np.ndarray:
    probe = LogisticRegression(max_iter=1000)
    probe.fit(hidden, labels)
    direction = probe.coef_[0]
    direction = direction / (np.linalg.norm(direction) + 1e-9)
    probe_acc = probe.score(hidden, labels)
    print(f"Linear probe on hidden activations: accuracy={probe_acc:.3f}, dim={direction.shape[0]}")
    return direction, probe_acc


def make_ablation_patch_fn(direction: torch.Tensor):
    """Return a patch_fn (works with either forward(patch_fn=...) or a hook)."""
    def patch_fn(h):
        proj = (h @ direction).unsqueeze(-1) * direction.unsqueeze(0)
        return h - proj
    return patch_fn


class ToySAE(nn.Module):
    """Real nn.Module SAE, trained with Adam. Same architecture as v1 (single
    ReLU hidden layer with L1 penalty on the code), now with proper
    optimizer + dead-feature tracking."""

    def __init__(self, input_dim, n_features=32, seed=2):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Linear(input_dim, n_features)
        self.dec = nn.Linear(n_features, input_dim, bias=False)

    def encode(self, h):
        return torch.relu(self.enc(h))

    def forward(self, h):
        code = self.encode(h)
        recon = self.dec(code)
        return recon, code

    def fit(self, H: torch.Tensor, epochs=800, lr=1e-3, l1=1e-4, verbose=False):
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            opt.zero_grad()
            recon, code = self.forward(H)
            recon_loss = ((recon - H) ** 2).sum(dim=1).mean()
            l1_loss = l1 * code.abs().sum(dim=1).mean()
            loss = recon_loss + l1_loss
            loss.backward()
            opt.step()
            if verbose and epoch % 200 == 0:
                print(f"  SAE epoch {epoch:4d}  recon_loss={recon_loss.item():.4f}  l1={l1_loss.item():.4f}")
        return self

    def dead_feature_frac(self, H: torch.Tensor, threshold=1e-6) -> float:
        with torch.no_grad():
            code = self.encode(H)
        return float((code.max(dim=0).values < threshold).float().mean())


def main():
    print("=" * 70)
    print("WORKSTREAM 1: Mechanistic Interpretability & Model Internals (v2)")
    print("=" * 70)

    X_train, y_train = to_tensors(*make_dataset(n_per_class=300, seed=0, split="train_concepts"))
    X_test, y_test = to_tensors(*make_dataset(n_per_class=150, seed=99, split="iid"))
    X_held, y_held = to_tensors(*make_dataset(n_per_class=80, seed=7, split="heldout_concepts"))

    model = MultimodalSafetyModel(input_dim=2 * EMBED_DIM)
    print("\n[1] Training base multimodal safety model (train-concept split)...")
    model.fit(X_train, y_train, epochs=300, lr=0.05)
    baseline = evaluate(model, X_test, y_test)
    print_report("baseline (unpatched, iid test)", baseline)
    held_baseline = evaluate(model, X_held, y_held)
    print_report("baseline (unpatched, HELD-OUT concepts)", held_baseline)

    print("\n[2] Probing hidden activations for an 'unsafe concept' direction...")
    with torch.no_grad():
        _, hidden_train = model.forward(X_train, return_hidden=True)
    direction_np, probe_acc = train_probe(hidden_train.numpy(), y_train.numpy())
    direction = torch.tensor(direction_np, dtype=torch.float32)

    print("\n[2b] Does the probe direction generalize to HELD-OUT concepts?")
    with torch.no_grad():
        _, hidden_held = model.forward(X_held, return_hidden=True)
    held_probe_acc = ((hidden_held.numpy() @ direction_np > 0).astype(int) == y_held.numpy()).mean()
    print(f"  Probe direction accuracy on held-out concepts: {held_probe_acc:.3f} "
          f"(vs {probe_acc:.3f} on training concepts)")
    if held_probe_acc > 0.75:
        print("  -> The direction generalizes to concepts never seen in training: this looks "
              "like a genuine 'unsafe' feature, not a memorized lookup of training concepts.")
    else:
        print("  -> The direction does NOT generalize well: likely memorizing training "
              "concepts rather than encoding a transferable notion of 'unsafe'.")

    print("\n[3] Activation patching via a real forward hook: ablating the probe direction...")
    patch_fn = make_ablation_patch_fn(direction)
    handle = register_activation_hook(model, patch_fn)
    patched = evaluate(model, X_test, y_test)  # hook is active, no need to pass patch_fn
    handle.remove()
    print_report("patched (direction ablated via hook)", patched)
    print(f"  -> Safety violation rate: {baseline['safety_violation_rate']:.3f} -> "
          f"{patched['safety_violation_rate']:.3f}")
    print(f"  -> False positive rate:   {baseline['false_positive_rate']:.3f} -> "
          f"{patched['false_positive_rate']:.3f}")

    print("\n[4] Training a sparse autoencoder (Adam, L1 penalty) on hidden activations...")
    with torch.no_grad():
        _, hidden_all = model.forward(torch.cat([X_train, X_test]), return_hidden=True)
    sae = ToySAE(input_dim=hidden_all.shape[1], n_features=32)
    sae.fit(hidden_all, epochs=800, lr=1e-3, l1=1e-4, verbose=True)
    dead_frac = sae.dead_feature_frac(hidden_all)
    print(f"  Dead feature fraction: {dead_frac:.1%} (features that never fire -- "
          "a standard SAE health metric; too high means wasted capacity, "
          "too low can mean features are too dense/polysemantic)")

    with torch.no_grad():
        codes = sae.encode(hidden_train)
    unsafe_mean = codes[y_train == 1].mean(dim=0)
    safe_mean = codes[y_train == 0].mean(dim=0)
    selectivity = unsafe_mean - safe_mean
    top_feature = int(torch.argmax(selectivity))
    print(f"\n  Most 'unsafe-selective' SAE feature: #{top_feature}  "
          f"(unsafe act={unsafe_mean[top_feature]:.3f}, safe act={safe_mean[top_feature]:.3f})")

    # --- plot: projection onto unsafe direction, safe vs unsafe, iid vs held-out ---
    with torch.no_grad():
        proj_test = (hidden_train @ direction).numpy()
        proj_held = (hidden_held @ direction).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, proj, y, title in [
        (axes[0], proj_test, y_train.numpy(), "Training concepts"),
        (axes[1], proj_held, y_held.numpy(), "Held-out concepts"),
    ]:
        ax.hist(proj[y == 0], bins=25, alpha=0.6, label="safe", color="#4C72B0")
        ax.hist(proj[y == 1], bins=25, alpha=0.6, label="unsafe", color="#C44E52")
        ax.set_title(title)
        ax.set_xlabel("projection onto probe direction")
        ax.legend()
    fig.suptitle("WS1: separation of the probe's unsafe direction, iid vs. held-out concepts")
    save_fig(fig, "ws1_probe_projection")
    plt.close(fig)

    save_result("workstream1_interpretability", {
        "baseline_iid": baseline,
        "baseline_heldout_concepts": held_baseline,
        "probe_accuracy_train_concepts": probe_acc,
        "probe_accuracy_heldout_concepts": float(held_probe_acc),
        "patched": patched,
        "sae_dead_feature_frac": dead_frac,
        "sae_top_unsafe_feature": top_feature,
    })


if __name__ == "__main__":
    main()
