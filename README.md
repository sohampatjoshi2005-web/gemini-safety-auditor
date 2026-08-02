# Gemini Safety Auditor -- Streamlit deployment

Single-file version of the Gemini Safety Auditor, merged for deployment on
Streamlit Community Cloud. All five audit phases, the embedding probe, the
offline fake-client demo mode, and the UI live in `streamlit_app.py` --
one file, no package layout needed.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on share.streamlit.io

1. Push `streamlit_app.py` and `requirements.txt` to a GitHub repo (the
   `findings/` directory is created automatically on first run -- you
   don't need to pre-create it, but it's fine if you do).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing at that repo, with `streamlit_app.py` as the main file.
3. **Do not** put your own `GEMINI_API_KEY` in Streamlit's "Secrets" for a
   public deployment unless you also add an access gate -- otherwise
   anyone who finds the URL can spend your quota. The app's default
   behavior already avoids this: each visitor pastes their own key in the
   sidebar, used only for that run, never written to disk or shared
   across sessions.

## Two modes, picked in the sidebar

- **Offline demo** -- a deterministic fake client, no API key, no cost, no
  network call. Proves the pipeline's plumbing works end-to-end. Says
  nothing about what a real Gemini model would actually do.
- **Real Gemini API** -- the visitor's own key, used only for that run.

## Why one file, and what changed from the multi-file version

The original multi-file version (`agent/`, `probes/` packages) used
module-level monkeypatching to swap in a fake client or a visitor's API
key, because each phase module imported `generate`/`embed` as its own
bound name. Collapsing everything into one file removes the need for
that: every phase function (`run_audit`, `run_eval_awareness_audit`, etc.)
now takes `use_fake` and `api_key` as **explicit parameters** instead.

This isn't just a simpler refactor -- it fixes a real caveat the original
README flagged: Streamlit Community Cloud's free tier runs one shared
process for every visitor. Global monkeypatching in that setup isn't
session-isolated between concurrent visitors. Passing `use_fake`/`api_key`
as plain function arguments has no shared mutable state, so it's safe
under concurrent sessions -- a genuine improvement, not just a smaller
diff.

## Honest scope (carried over, still true)

- The probe is fit on Gemini's public embeddings API output, not real
  hidden-layer activations -- a real but coarser, correlational signal.
- Phase 1's "attack" is a black-box heuristic rephrasing search, not an
  optimized adversarial (GCG-style) suffix attack -- no gradient access
  to a hosted model. Same limitation applies to Phase 4's image condition.
- The seed dataset is 12 hand-written examples -- enough to demo the
  pipeline, not a validated classifier.
- Every phase is a small spot-check on a handful of prompts, run once
  each. Any shift/disagreement found is worth a closer look, not a
  confirmed, statistically powered finding.

## What got folded in vs. dropped

- `main.py` / `smoke_test.py`'s CLI entry points aren't included here --
  the sidebar's **Offline demo** mode covers the same purpose (proving the
  pipeline runs without a real key) interactively. If you still want a
  standalone CLI smoke test for local dev outside Streamlit, ask and it's
  a quick add-on.
- All five phases, the probe, and the seed dataset are unchanged in
  behavior from the multi-file version -- verified by running every phase
  in offline mode before this was packaged.
