"""
Binds a caller-supplied API key into every phase module's already-imported
`generate`/`embed` names for the duration of a `with` block, via
functools.partial -- same monkeypatch pattern as agent/fake_client.py.

Why not just `os.environ["GEMINI_API_KEY"] = key`? A Streamlit deployment
runs one process shared by every visitor's session (each in its own
thread/script-run). Setting a process-wide env var would let one visitor's
key leak into another concurrent session's calls. Binding the key directly
into the call arguments keeps it scoped to the current run.
"""

from functools import partial
from contextlib import contextmanager

from . import gemini_client


@contextmanager
def use_session_api_key(api_key: str):
    import agent.orchestrator as orchestrator
    import agent.eval_awareness as eval_awareness
    import agent.injection_probe as injection_probe
    import agent.multimodal_probe as multimodal_probe
    import agent.cross_tier as cross_tier

    modules = (orchestrator, eval_awareness, injection_probe, multimodal_probe, cross_tier)
    originals = {}
    bound_generate = partial(gemini_client.generate, api_key=api_key)
    bound_embed = partial(gemini_client.embed, api_key=api_key)

    for mod in modules:
        originals[mod] = {"generate": mod.generate}
        mod.generate = bound_generate
        if hasattr(mod, "embed"):
            originals[mod]["embed"] = mod.embed
            mod.embed = bound_embed

    try:
        yield
    finally:
        for mod, orig in originals.items():
            mod.generate = orig["generate"]
            if "embed" in orig:
                mod.embed = orig["embed"]
