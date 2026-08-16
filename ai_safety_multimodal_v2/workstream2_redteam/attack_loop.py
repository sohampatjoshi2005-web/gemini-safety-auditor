"""
workstream2_redteam/attack_loop.py (v2)

Upgrade over v1: this is now a REAL PGD (Projected Gradient Descent) attack
using torch.autograd, not gradient-free hill-climbing. This is the actual
algorithm used in adversarial ML research (Madry et al. 2017), not a
methodology stand-in for it -- the only thing that's still "toy" is the
model being attacked, not the attack algorithm.

SAFETY NOTE: this only ever attacks our own synthetic toy classifier on
synthetic embeddings. No real jailbreak content, real model weaknesses, or
usable exploit techniques are produced here.
"""

import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "shared"))
from dataset import make_dataset, to_tensors, EMBED_DIM  # noqa: E402
from model import MultimodalSafetyModel  # noqa: E402
from eval_harness import evaluate, print_report  # noqa: E402
from utils import save_result, save_fig  # noqa: E402


def pgd_attack(model, x: torch.Tensor, eps=1.5, alpha=0.1, n_steps=100):
    """
    Real PGD: x is a single (input_dim,) example. Only the IMAGE half
    (last EMBED_DIM entries) is perturbed -- attacker controls the image,
    not the text, same threat model as v1. eps is the L2 budget on the
    perturbation, alpha is the step size, exactly matching the structure of
    a real adversarial-robustness PGD loop.
    """
    text_part = x[:EMBED_DIM].clone()
    image_part = x[EMBED_DIM:].clone()
    delta = torch.zeros_like(image_part, requires_grad=True)

    best_p = model.predict_proba(x.unsqueeze(0)).item()
    best_delta = torch.zeros_like(image_part)

    for _ in range(n_steps):
        adv_image = image_part + delta
        adv_x = torch.cat([text_part, adv_image]).unsqueeze(0)
        logit = model.forward(adv_x)
        p = torch.sigmoid(logit)
        # minimize P(unsafe): loss = p (we want to push p down)
        loss = p.sum()
        grad = torch.autograd.grad(loss, delta)[0]

        with torch.no_grad():
            delta -= alpha * grad.sign()  # signed gradient step (standard PGD)
            norm = delta.norm()
            if norm > eps:
                delta *= eps / norm  # project back into the L2 ball (the "P" in PGD)

            p_now = model.predict_proba(torch.cat([text_part, image_part + delta]).unsqueeze(0)).item()
            if p_now < best_p:
                best_p = p_now
                best_delta = delta.clone()
        delta.requires_grad_(True)

    x_adv = torch.cat([text_part, image_part + best_delta])
    success = best_p < 0.5
    return x_adv, best_p, success


def run_attack_campaign(model, X_unsafe: torch.Tensor, eps=1.5, alpha=0.1, n_steps=100):
    successes = 0
    adv_examples = []
    for i in range(len(X_unsafe)):
        x_adv, final_p, success = pgd_attack(model, X_unsafe[i], eps=eps, alpha=alpha, n_steps=n_steps)
        successes += int(success)
        adv_examples.append((x_adv.detach(), success))
    asr = successes / len(X_unsafe)
    return asr, adv_examples


def adversarial_finetune(model, X_unsafe, adv_examples, epochs=150, lr=0.05):
    adv_X = torch.stack([x for x, success in adv_examples if success]) if any(s for _, s in adv_examples) else None
    if adv_X is None or len(adv_X) == 0:
        print("  No successful attacks to train against.")
        return model
    adv_y = torch.ones(len(adv_X), dtype=torch.long)
    X_combined = torch.cat([X_unsafe, adv_X])
    y_combined = torch.cat([torch.ones(len(X_unsafe), dtype=torch.long), adv_y])
    model.fit(X_combined, y_combined, epochs=epochs, lr=lr)
    return model


def main():
    print("=" * 70)
    print("WORKSTREAM 2: AI Security & Frontier Red Team (v2 -- real PGD)")
    print("=" * 70)

    X_train, y_train = to_tensors(*make_dataset(n_per_class=300, seed=0))
    X_test, y_test = to_tensors(*make_dataset(n_per_class=150, seed=99))

    model = MultimodalSafetyModel(input_dim=2 * EMBED_DIM)
    print("\n[1] Training base multimodal safety classifier...")
    model.fit(X_train, y_train, epochs=300, lr=0.05)
    baseline = evaluate(model, X_test, y_test)
    print_report("baseline", baseline)

    X_unsafe = X_test[y_test == 1]
    print(f"\n[2] Running REAL PGD attack campaign on {len(X_unsafe)} unsafe-labeled inputs "
          f"(L2 budget=1.5, image embedding only)...")
    asr, adv_examples = run_attack_campaign(model, X_unsafe, eps=1.5, alpha=0.1, n_steps=100)
    print(f"  Attack Success Rate (ASR): {asr:.1%}")

    # sweep attack budget for a plot -- shows the actual robustness curve
    print("\n[2b] Sweeping perturbation budget (eps) for an ASR-vs-budget curve...")
    eps_values = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    asr_by_eps = []
    for eps in eps_values:
        a, _ = run_attack_campaign(model, X_unsafe[:40], eps=eps, alpha=0.1, n_steps=60)
        asr_by_eps.append(a)
        print(f"  eps={eps:.2f}  ASR={a:.1%}")

    print("\n[3] Adversarial fine-tuning as a mitigation...")
    model_defended = MultimodalSafetyModel(input_dim=2 * EMBED_DIM, seed=11)
    model_defended.fit(X_train, y_train, epochs=300, lr=0.05)
    model_defended = adversarial_finetune(model_defended, X_unsafe, adv_examples)

    defended_metrics = evaluate(model_defended, X_test, y_test)
    print_report("defended (post adversarial fine-tune)", defended_metrics)

    asr_after, _ = run_attack_campaign(model_defended, X_unsafe, eps=1.5, alpha=0.1, n_steps=100)
    print(f"  ASR after defense: {asr_after:.1%}  (was {asr:.1%} before)")
    print("  -> Watch false_positive_rate above -- adversarial training that overcorrects "
          "raises false positives, the usability tradeoff every red-team mitigation must report.")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(eps_values, [a * 100 for a in asr_by_eps], marker="o", color="#C44E52")
    ax.set_xlabel("perturbation budget (L2 norm)")
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("WS2: PGD attack success rate vs. perturbation budget")
    ax.grid(alpha=0.3)
    save_fig(fig, "ws2_asr_vs_budget")
    plt.close(fig)

    save_result("workstream2_redteam", {
        "baseline": baseline,
        "asr_before_defense": asr,
        "asr_after_defense": asr_after,
        "defended_metrics": defended_metrics,
        "asr_vs_eps": dict(zip(eps_values, asr_by_eps)),
    })


if __name__ == "__main__":
    main()
