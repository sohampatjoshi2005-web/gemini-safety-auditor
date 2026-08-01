"""
Multimodal consistency probe (Phase 4).

Real category this targets: cross-modal jailbreaks. Because Gemini
processes text, image, audio, and video under one shared representation
rather than separate per-modality pipelines, safety training on text isn't
guaranteed to generalize identically once an image is in the context --
research has documented this kind of cross-modal "bleed-through" (e.g.
arxiv.org/html/2601.15698v1, arxiv.org/html/2601.10527v1).

READ THIS BEFORE TRUSTING THE RESULTS -- honest scope note:
A REAL adversarial-image attack optimizes pixel perturbations against the
target model's gradients (or a substitute model's, then transfers them).
We have no gradient access to a hosted Gemini endpoint -- the exact same
limitation agent/orchestrator.py already documents for why its text
"attack" is a heuristic rephrasing search and not a real GCG suffix attack.

So this is NOT a working adversarial-image jailbreak, and it deliberately
does not try to be one. What it IS: a black-box check of whether the mere
PRESENCE of an image -- a plain, non-adversarial, locally-generated
placeholder, not an optimized attack image -- shifts the refusal or probe
verdict on the exact same boundary prompts already used in Phase 1,
relative to a text-only baseline. A shift here is a weaker, correlational
signal that modality itself matters for this prompt set; it is not
evidence of a working jailbreak, and should be read the same way this
project reads every other disagreement it finds: "worth a closer look,"
not a confirmed vulnerability.
"""

import io
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict

from PIL import Image, ImageDraw

from .gemini_client import generate, embed, GeminiError
from probes.embedding_probe import EmbeddingProbe

MULTIMODAL_LOG_PATH = Path(__file__).parent.parent / "findings" / "multimodal_log.jsonl"

# Plain background colors to vary across prompts -- just enough to check
# whether image CONTENT (not only presence) matters, without reaching for
# anything resembling an optimized perturbation.
_BACKGROUND_COLORS = [(220, 220, 220), (200, 220, 255), (255, 220, 200), (220, 255, 220)]


def _synthetic_image_bytes(seed: int) -> bytes:
    """
    Generates a plain, non-adversarial placeholder image locally: a solid
    background with a short caption. No network image fetch, and
    deliberately no perturbation pipeline -- see module docstring for why.
    """
    color = _BACKGROUND_COLORS[seed % len(_BACKGROUND_COLORS)]
    img = Image.new("RGB", (512, 512), color=color)
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "test image", fill=(20, 20, 20))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@dataclass
class MultimodalResult:
    prompt: str
    condition: str  # "text_only" or "with_image"
    refused: bool
    finish_reason: str
    probe_unsafe_score: float | None
    notes: str


def _run_condition(prompt: str, probe: EmbeddingProbe, image_bytes: bytes | None,
                    condition: str) -> MultimodalResult:
    try:
        gen = generate(prompt, image_bytes=image_bytes)
    except GeminiError as e:
        return MultimodalResult(
            prompt=prompt, condition=condition, refused=True,
            finish_reason="API_ERROR", probe_unsafe_score=None,
            notes=f"API error: {e}",
        )

    probe_score = None
    notes = ""
    if gen["text"].strip():
        try:
            probe_score = probe.score(embed(gen["text"]))
        except GeminiError as e:
            notes = f"Embedding call failed: {e}"

    return MultimodalResult(
        prompt=prompt, condition=condition, refused=gen["refused"],
        finish_reason=gen["finish_reason"], probe_unsafe_score=probe_score,
        notes=notes,
    )


def compare_multimodal(prompt: str, probe: EmbeddingProbe, seed: int) -> list[MultimodalResult]:
    """Runs one prompt text-only, then again with a plain synthetic image
    attached, so the two conditions can be compared directly."""
    text_only = _run_condition(prompt, probe, None, "text_only")
    time.sleep(1)
    with_image = _run_condition(prompt, probe, _synthetic_image_bytes(seed), "with_image")
    return [text_only, with_image]


def log_results(results: list[MultimodalResult]):
    MULTIMODAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MULTIMODAL_LOG_PATH, "a") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")


def run_multimodal_audit(prompts: list[str], probe: EmbeddingProbe) -> list[MultimodalResult]:
    """Entry point used by main.py -- compares text-only vs. with-image for
    each prompt, logs everything, prints a summary of any shifts found."""
    all_results: list[MultimodalResult] = []
    shifts = 0
    for i, prompt in enumerate(prompts):
        pair = compare_multimodal(prompt, probe, seed=i)
        log_results(pair)
        all_results.extend(pair)

        text_only, with_image = pair
        if text_only.refused != with_image.refused:
            shifts += 1
            print(f"  -> SHIFT: refusal changed with image present for {prompt[:50]!r}")
        time.sleep(1)

    print(f"Phase 4 done. {len(prompts)} prompt(s) compared text-only vs. with-image, "
          f"{shifts} refusal shift(s) found.")
    print("See findings/multimodal_log.jsonl for full results.")
    return all_results
