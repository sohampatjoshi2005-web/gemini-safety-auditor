"""
Cross-tier speed-vs-safety probe (Phase 5).

Question this closes out from the README roadmap: does picking the
cheaper/faster generation tier trade away safety behavior, not just
latency? This runs the same boundary prompts through two Gemini tiers --
config.GENERATION_MODEL as the "fast" tier and config.GENERATION_MODEL_CAPABLE
as the "capable" tier -- and compares wall-clock latency alongside the same
behavioral + probe verdicts Phase 1 already uses.

Honest scope note: two model tiers on a handful of prompts, run once each,
is a spot-check, not a statistically powered benchmark. Treat any
difference found as "worth a closer look with more prompts and more
repeats," not a confirmed regression -- and note that latency on a hosted
API includes network/queue variance outside the model's own compute time,
which averages out over more calls but won't on a single run.
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict

from .gemini_client import generate, embed, GeminiError
from .config import GENERATION_MODEL, GENERATION_MODEL_CAPABLE
from probes.embedding_probe import EmbeddingProbe

CROSS_TIER_LOG_PATH = Path(__file__).parent.parent / "findings" / "cross_tier_log.jsonl"


@dataclass
class CrossTierResult:
    prompt: str
    model: str
    tier: str  # "fast" or "capable"
    latency_seconds: float
    refused: bool
    finish_reason: str
    probe_unsafe_score: float | None


def _run_tier(prompt: str, model: str, tier: str, probe: EmbeddingProbe) -> CrossTierResult:
    start = time.monotonic()
    try:
        gen = generate(prompt, model=model)
    except GeminiError:
        elapsed = time.monotonic() - start
        return CrossTierResult(
            prompt=prompt, model=model, tier=tier, latency_seconds=elapsed,
            refused=True, finish_reason="API_ERROR", probe_unsafe_score=None,
        )
    elapsed = time.monotonic() - start

    probe_score = None
    if gen["text"].strip():
        try:
            probe_score = probe.score(embed(gen["text"]))
        except GeminiError:
            pass  # non-fatal -- leave probe_score as None

    return CrossTierResult(
        prompt=prompt, model=model, tier=tier, latency_seconds=elapsed,
        refused=gen["refused"], finish_reason=gen["finish_reason"],
        probe_unsafe_score=probe_score,
    )


def compare_tiers(prompt: str, probe: EmbeddingProbe) -> list[CrossTierResult]:
    fast = _run_tier(prompt, GENERATION_MODEL, "fast", probe)
    time.sleep(1)
    capable = _run_tier(prompt, GENERATION_MODEL_CAPABLE, "capable", probe)
    return [fast, capable]


def log_results(results: list[CrossTierResult]):
    CROSS_TIER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CROSS_TIER_LOG_PATH, "a") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")


def run_cross_tier_audit(prompts: list[str], probe: EmbeddingProbe) -> list[CrossTierResult]:
    """Entry point used by main.py -- compares fast vs. capable tier for
    each prompt, logs everything, prints a summary of any regressions found."""
    all_results: list[CrossTierResult] = []
    regressions = 0
    for prompt in prompts:
        pair = compare_tiers(prompt, probe)
        log_results(pair)
        all_results.extend(pair)

        fast, capable = pair
        if fast.refused != capable.refused:
            regressions += 1
            print(f"  -> SAFETY REGRESSION CANDIDATE: refusal differs between tiers for {prompt[:50]!r}")
        speedup = (capable.latency_seconds / fast.latency_seconds) if fast.latency_seconds > 0 else float("nan")
        print(f"  fast={fast.latency_seconds:.2f}s capable={capable.latency_seconds:.2f}s "
              f"({speedup:.1f}x) for {prompt[:40]!r}")
        time.sleep(1)

    print(f"Phase 5 done. {len(prompts)} prompt(s) compared across tiers, "
          f"{regressions} safety-regression candidate(s) found.")
    print("See findings/cross_tier_log.jsonl for full results.")
    return all_results
