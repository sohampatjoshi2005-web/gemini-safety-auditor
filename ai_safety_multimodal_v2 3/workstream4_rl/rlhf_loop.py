"""
workstream4_rl/rlhf_loop.py (v2)

Same design as v1 (REINFORCE on a Gaussian generator policy, reward = WS1's
safety classifier, cross-check against WS1's probe direction to detect
reward hacking), now using torch.distributions.Normal + torch.optim.Adam
for the policy gradient update instead of hand-derived REINFORCE gradients,
and logging a reward curve + before/after diversity plot.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "shared"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "workstream1_interpretability"))
from dataset import make_dataset, to_tensors, embed_text, EMBED_DIM  # noqa: E402
from model import MultimodalSafetyModel  # noqa: E402
from probe_and_patch import train_probe  # noqa: E402
from utils import save_result, save_fig  # noqa: E402


class GeneratorPolicy(nn.Module):
    """Gaussian policy over image embeddings, conditioned on a fixed text prompt."""

    def __init__(self, text_dim=EMBED_DIM, out_dim=EMBED_DIM, seed=3, std=0.6):
        super().__init__()
        torch.manual_seed(seed)
        self.linear = nn.Linear(text_dim, out_dim)
        self.log_std = nn.Parameter(torch.log(torch.tensor(std)), requires_grad=False)

    def dist(self, text_emb):
        mu = self.linear(text_emb)
        return Normal(mu, self.log_std.exp())

    def sample(self, text_emb, n):
        d = self.dist(text_emb)
        samples = d.sample((n,))
        return samples, d


def mean_pairwise_distance(X, max_pairs=2000, seed=0):
    r = np.random.default_rng(seed)
    n = len(X)
    idx_a = r.integers(0, n, size=max_pairs)
    idx_b = r.integers(0, n, size=max_pairs)
    return float(np.linalg.norm(X[idx_a] - X[idx_b], axis=1).mean())


def main():
    print("=" * 70)
    print("WORKSTREAM 4: Reinforcement Learning Alignment (v2 -- torch REINFORCE)")
    print("=" * 70)

    X_train, y_train = to_tensors(*make_dataset(n_per_class=300, seed=0))
    reward_model = MultimodalSafetyModel(input_dim=2 * EMBED_DIM)
    print("\n[1] Training reward model (safety classifier)...")
    reward_model.fit(X_train, y_train, epochs=300, lr=0.05)
    with torch.no_grad():
        _, hidden_train = reward_model.forward(X_train, return_hidden=True)
    unsafe_direction_np, _ = train_probe(hidden_train.numpy(), y_train.numpy())
    unsafe_direction = torch.tensor(unsafe_direction_np, dtype=torch.float32)

    text_emb = torch.tensor(embed_text("a weapon"), dtype=torch.float32)

    def project_onto_unsafe_direction(image_emb_batch: torch.Tensor) -> float:
        x = torch.cat([text_emb.unsqueeze(0).expand(len(image_emb_batch), -1), image_emb_batch], dim=1)
        with torch.no_grad():
            _, hidden = reward_model.forward(x, return_hidden=True)
        return (hidden @ unsafe_direction).mean().item()

    def reward_fn(image_emb_batch: torch.Tensor) -> torch.Tensor:
        x = torch.cat([text_emb.unsqueeze(0).expand(len(image_emb_batch), -1), image_emb_batch], dim=1)
        with torch.no_grad():
            p_unsafe = torch.sigmoid(reward_model.forward(x))
        return 1.0 - 2.0 * p_unsafe  # in [-1, 1]: high = confidently safe

    policy = GeneratorPolicy(seed=3)
    opt = torch.optim.Adam(policy.parameters(), lr=0.02)

    print("\n[2] Pre-RL generation stats (untrained policy)...")
    with torch.no_grad():
        pre_samples, _ = policy.sample(text_emb, n=500)
    pre_rewards = reward_fn(pre_samples)
    pre_violation_rate = (pre_rewards < 0).float().mean().item()
    pre_diversity = mean_pairwise_distance(pre_samples.numpy())
    pre_proj = project_onto_unsafe_direction(pre_samples)
    print(f"  Safety violation rate: {pre_violation_rate:.3f}")
    print(f"  Output diversity (mean pairwise L2): {pre_diversity:.3f}")
    print(f"  Mean projection onto WS1's unsafe direction: {pre_proj:.3f}")

    print("\n[3] Running REINFORCE RL fine-tuning loop (torch.optim.Adam)...")
    n_iters, batch_size = 400, 64
    reward_curve = []
    for it in range(n_iters):
        samples, dist = policy.sample(text_emb, n=batch_size)
        rewards = reward_fn(samples)
        norm_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        log_probs = dist.log_prob(samples).sum(dim=1)
        loss = -(log_probs * norm_rewards).mean()  # REINFORCE loss, real autograd
        opt.zero_grad()
        loss.backward()
        opt.step()
        reward_curve.append(rewards.mean().item())
        if it % 100 == 0:
            print(f"  iter {it:4d}  mean_reward={rewards.mean().item():+.3f}")

    print("\n[4] Post-RL generation stats...")
    with torch.no_grad():
        post_samples, _ = policy.sample(text_emb, n=500)
    post_rewards = reward_fn(post_samples)
    post_violation_rate = (post_rewards < 0).float().mean().item()
    post_diversity = mean_pairwise_distance(post_samples.numpy())
    post_proj = project_onto_unsafe_direction(post_samples)
    print(f"  Safety violation rate: {pre_violation_rate:.3f} -> {post_violation_rate:.3f}")
    print(f"  Output diversity (mean pairwise L2): {pre_diversity:.3f} -> {post_diversity:.3f}")
    print(f"  Mean projection onto WS1's unsafe direction: {pre_proj:.3f} -> {post_proj:.3f}")

    diversity_drop = (pre_diversity - post_diversity) / pre_diversity * 100
    print(f"\n  -> Diversity changed by {diversity_drop:+.1f}% after RL alignment (mode-collapse check).")

    reward_hacking = post_proj >= pre_proj
    if not reward_hacking:
        print(f"  -> Projection DECREASED ({pre_proj:.3f} -> {post_proj:.3f}): RL moved generations "
              "genuinely away from the unsafe direction -- a reassuring sign.")
    else:
        print(f"  -> Projection INCREASED ({pre_proj:.3f} -> {post_proj:.3f}) even though the "
              f"classifier's violation rate dropped to {post_violation_rate:.3f}: reward hacking near "
              "the decision boundary -- interpretability caught something the safety metric alone missed.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(reward_curve, color="#4C72B0", linewidth=1)
    axes[0].set_xlabel("RL iteration")
    axes[0].set_ylabel("mean batch reward")
    axes[0].set_title("WS4: REINFORCE reward curve")
    axes[0].grid(alpha=0.3)

    labels = ["violation rate", "diversity", "unsafe-dir projection"]
    pre_vals = [pre_violation_rate, pre_diversity, pre_proj]
    post_vals = [post_violation_rate, post_diversity, post_proj]
    x = np.arange(len(labels))
    axes[1].bar(x - 0.18, pre_vals, width=0.35, label="pre-RL", color="#8C8C8C")
    axes[1].bar(x + 0.18, post_vals, width=0.35, label="post-RL", color="#55A868")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15)
    axes[1].set_title("WS4: pre/post-RL comparison")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")
    save_fig(fig, "ws4_rl_alignment")
    plt.close(fig)

    save_result("workstream4_rl", {
        "pre": {"violation_rate": pre_violation_rate, "diversity": pre_diversity, "unsafe_proj": pre_proj},
        "post": {"violation_rate": post_violation_rate, "diversity": post_diversity, "unsafe_proj": post_proj},
        "diversity_drop_pct": diversity_drop,
        "reward_hacking_detected": reward_hacking,
    })


if __name__ == "__main__":
    main()
