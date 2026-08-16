"""
shared/utils.py

Shared helpers so every workstream script (a) saves its numeric results to
results/summary.json in a consistent shape, and (b) saves at least one plot
to results/. This is what makes the project "showcase-able" -- a reviewer
should be able to open results/ and see the story without reading logs.
"""

import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
SUMMARY_PATH = os.path.join(RESULTS_DIR, "summary.json")


def save_result(workstream: str, data: dict):
    """Merge `data` under summary[workstream] and write results/summary.json."""
    summary = {}
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH) as f:
            try:
                summary = json.load(f)
            except json.JSONDecodeError:
                summary = {}
    summary[workstream] = data
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  [saved results/summary.json -> '{workstream}']")


def save_fig(fig, name: str):
    path = os.path.join(RESULTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"  [saved {os.path.relpath(path)}]")
    return path
