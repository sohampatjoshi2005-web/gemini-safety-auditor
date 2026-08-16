"""
workstream3_systems/benchmark.py (v2)

Same methodology as v1 (P50/P90 latency + throughput across batch sizes,
comparing base model vs. +classifier vs. +classifier+steering), now run
against the real PyTorch model with torch.no_grad() inference (matching how
a real serving stack would run this), and saving a latency-vs-batch-size
plot instead of only printing a table.
"""

import sys
import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "shared"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "workstream1_interpretability"))
from dataset import make_dataset, to_tensors, EMBED_DIM  # noqa: E402
from model import MultimodalSafetyModel  # noqa: E402
from probe_and_patch import train_probe, make_ablation_patch_fn  # noqa: E402
from utils import save_result, save_fig  # noqa: E402


def time_forward(fn, X, n_repeats=30):
    latencies = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn(X)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
    return latencies


def summarize(name, latencies, batch_size):
    arr = np.array(latencies)
    p50, p90 = np.percentile(arr, 50), np.percentile(arr, 90)
    throughput = batch_size / (p50 / 1000.0)
    print(f"  {name:38s} batch={batch_size:4d}  P50={p50:7.3f}ms  P90={p90:7.3f}ms  "
          f"throughput={throughput:9.1f} ex/s")
    return {"name": name, "batch_size": batch_size, "p50_ms": float(p50),
             "p90_ms": float(p90), "throughput": float(throughput)}


def main():
    print("=" * 70)
    print("WORKSTREAM 3: ML Systems & Performance (v2)")
    print("=" * 70)

    X_train, y_train = to_tensors(*make_dataset(n_per_class=300, seed=0))
    model = MultimodalSafetyModel(input_dim=2 * EMBED_DIM)
    model.fit(X_train, y_train, epochs=300, lr=0.05)
    model.eval()

    with torch.no_grad():
        _, hidden_train = model.forward(X_train, return_hidden=True)
    direction_np, _ = train_probe(hidden_train.numpy(), y_train.numpy())
    direction = torch.tensor(direction_np, dtype=torch.float32)
    patch_fn = make_ablation_patch_fn(direction)

    guard = MultimodalSafetyModel(input_dim=2 * EMBED_DIM, hidden_dim=16, seed=5)  # lightweight guard model

    def config_base(X):
        with torch.no_grad():
            return model.forward(X)

    def config_with_classifier(X):
        with torch.no_grad():
            out = model.forward(X)
            _ = guard.forward(X)
        return out

    def config_with_classifier_and_steering(X):
        with torch.no_grad():
            out = model.forward(X, patch_fn=patch_fn)
            _ = guard.forward(X)
        return out

    configs = [
        ("(a) base model only", config_base),
        ("(b) base + safety classifier", config_with_classifier),
        ("(c) base + classifier + steering hook", config_with_classifier_and_steering),
    ]

    results = []
    batch_sizes = [1, 32, 256]
    print("\nBenchmarking across batch sizes (30 repeats each, after 5 warmup calls)...\n")
    for batch_size in batch_sizes:
        X_bench, _ = to_tensors(*make_dataset(n_per_class=max(batch_size // 2, 1), seed=42))
        X_bench = X_bench[:batch_size]
        for name, fn in configs:
            for _ in range(5):
                fn(X_bench)
            latencies = time_forward(fn, X_bench, n_repeats=30)
            results.append(summarize(name, latencies, batch_size))
        print()

    print("Summary -- overhead of guardrails relative to base model, at batch=256:")
    base_p50 = next(r["p50_ms"] for r in results if r["name"].startswith("(a)") and r["batch_size"] == 256)
    for r in results:
        if r["batch_size"] == 256:
            r["overhead_pct_vs_base"] = (r["p50_ms"] - base_p50) / base_p50 * 100
            print(f"  {r['name']:38s} +{r['overhead_pct_vs_base']:5.1f}% latency vs. base model")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, _ in configs:
        xs = [r["batch_size"] for r in results if r["name"] == name]
        ys = [r["p50_ms"] for r in results if r["name"] == name]
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch size")
    ax.set_ylabel("P50 latency (ms)")
    ax.set_title("WS3: latency vs. batch size, base model vs. guardrail configs")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    save_fig(fig, "ws3_latency_vs_batch")
    plt.close(fig)

    print("\n  -> This is the table/plot a serving/infra team would use to decide whether a "
          "safety intervention is cheap enough to run on every request, or whether it "
          "needs a cheap-classifier-first early-exit strategy.")

    save_result("workstream3_systems", {"results": results})


if __name__ == "__main__":
    main()
