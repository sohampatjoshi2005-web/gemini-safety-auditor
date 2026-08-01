"""
Deterministic fake Gemini client -- factored out of smoke_test.py so both
the CLI smoke test and the Streamlit app's "offline demo" mode share the
exact same fake generate()/embed() behavior instead of drifting apart.

No network, no API key required. This proves the pipeline's plumbing
works end-to-end -- it says nothing about what a real Gemini model would
actually do. See smoke_test.py's module docstring for the full caveat.
"""

import random
import time as time_module
from contextlib import contextmanager

EMB_DIM = 16
UNSAFE_MARKERS = ("exploit", "bypass", "synthesize", "manipulate", "restricted", "fake identification")


def fake_embed(text: str, retries: int = 3):
    """Deterministic fake embedding: hashes text into a fixed-size vector so
    the same text always embeds the same way, and unsafe-flavored seed text
    lands on a different side of the probe than safe-flavored text."""
    random.seed(hash(text) % (2**32))
    vec = [random.uniform(-1, 1) for _ in range(EMB_DIM)]
    if any(m in text.lower() for m in UNSAFE_MARKERS):
        vec[0] += 3.0
    return vec


def fake_generate(prompt: str, image_bytes=None, image_mime="image/jpeg",
                   retries: int = 3, model: str | None = None):
    """Deterministic fake generation covering three scripted branches so
    every downstream code path (refusal, canary-follow, normal answer)
    gets exercised at least once."""
    if "SYSTEM OVERRIDE" in prompt:
        random.seed(hash(prompt) % (2**32))
        if random.random() < 0.5:
            marker = "exactly this string and nothing else: "
            idx = prompt.find(marker)
            canary = prompt[idx + len(marker):].split("]")[0].strip() if idx != -1 else "UNKNOWN"
            return {"text": canary, "refused": False, "finish_reason": "STOP", "raw": {}}
        return {"text": "Sure -- here's a plain summary of the content you shared.",
                "refused": False, "finish_reason": "STOP", "raw": {}}

    if "bypass the security system" in prompt.lower():
        return {"text": "", "refused": True, "finish_reason": "SAFETY", "raw": {}}

    return {"text": f"Here is a helpful, on-topic answer about: {prompt[:40]}",
            "refused": False, "finish_reason": "STOP", "raw": {}}


def _patch_all():
    """`from .gemini_client import generate, embed` binds a new local name
    in each importing module, so the fake has to be installed on each
    module's own namespace -- reassigning gemini_client.generate afterward
    wouldn't reach any of these already-bound references."""
    import agent.orchestrator as orchestrator
    import agent.eval_awareness as eval_awareness
    import agent.injection_probe as injection_probe
    import agent.multimodal_probe as multimodal_probe
    import agent.cross_tier as cross_tier

    modules = (orchestrator, eval_awareness, injection_probe, multimodal_probe, cross_tier)
    originals = {}
    for mod in modules:
        originals[mod] = {"generate": mod.generate}
        mod.generate = fake_generate
        if hasattr(mod, "embed"):
            originals[mod]["embed"] = mod.embed
            mod.embed = fake_embed

    orig_sleep = time_module.sleep
    time_module.sleep = lambda *_a, **_k: None
    return originals, orig_sleep


def _unpatch_all(originals, orig_sleep):
    for mod, orig in originals.items():
        mod.generate = orig["generate"]
        if "embed" in orig:
            mod.embed = orig["embed"]
    time_module.sleep = orig_sleep


@contextmanager
def use_fake_client():
    """Context manager: patches every phase module to use the deterministic
    fake client for the duration of the `with` block, then restores the
    real gemini_client bindings.

    NOTE: this patches module-level attributes process-wide, same as
    smoke_test.py's patch_all(). Fine for a single-user CLI run or a
    single-user Streamlit session; if you deploy this app for concurrent
    multi-user traffic, this global patch is not thread-isolated between
    sessions running at the same instant.
    """
    originals, orig_sleep = _patch_all()
    try:
        yield
    finally:
        _unpatch_all(originals, orig_sleep)
