# Gemini Safety Auditor

An agent that audits a real Gemini model using a **plan → act → verify → retry**
loop, checking two signals at once for every probe prompt:

1. **Behavioral** — did the model refuse, based on its own `finishReason`?
2. **Causal-proxy** — does a linear probe trained on Gemini's embeddings think
   the *output* still looks unsafe-like, regardless of whether it refused?

The interesting result is when these two **disagree** — a case where a
purely behavioral audit would call the response "safe," but the embedding
probe still flags it. That disagreement is the project's core finding.

## Why this, and what it extends

Google DeepMind's own [Gram framework](https://arxiv.org/abs/2605.30322)
does automated alignment auditing on Gemini models via simulated agentic
environments and behavioral observation — did the model sabotage, yes or no.
That's a real and effective approach, but it's still an output-level check.

This project adds a second, cheaper signal underneath it: instead of (or
alongside) watching what the model *does*, check whether its *output
representation* still carries the "unsafe" direction a linear probe learned,
even when the behavioral check passed. This is explicitly a small,
honest instance of the same distinction the field's specification-gaming
literature is about — "looks safe" vs. "is safe" — built at a scale one
developer can run and verify, not a claim of closing that gap.

## What this is honestly *not*

- **Not a real activation probe.** A hosted API gives no access to hidden
  layers or gradients. The probe here is fit on Gemini's public embeddings
  API output — a real but coarser, correlational signal, not the causal
  ablation test a white-box probe would allow.
- **Not a gradient-based (GCG-style) attack.** The suffix search in
  `agent/orchestrator.py` is black-box and heuristic — it tries a few
  rephrased/escalated variants, not an optimized adversarial suffix. Real
  gradient-based attacks need weight access this project doesn't have.
  The same limitation applies to the image side of `agent/multimodal_probe.py`
  (Phase 4) — see that module's docstring.
- **Not a claim of novel research.** The seed dataset is 12 hand-written
  examples — enough to demo the pipeline end-to-end, not a validated
  classifier. Any disagreement case found should be read as "worth a closer
  look," not as a confirmed finding.

## Setup

```bash
git clone <this repo>
cd gemini_safety_auditor
pip install -r requirements.txt
export GEMINI_API_KEY="your-key-here"   # from https://aistudio.google.com
python3 main.py
```

## Streamlit app

`streamlit_app.py` is a second front end over the same `agent/` and
`probes/` packages -- a UI instead of a CLI. Nothing in the underlying
pipeline depends on Streamlit; this file only imports and orchestrates it.

```bash
streamlit run streamlit_app.py
```

It has two modes, chosen in the sidebar:
- **Offline demo** -- the same deterministic fake client `smoke_test.py`
  uses. No API key, no cost, no network call. Good for letting someone
  click through the pipeline's shape without spending your quota.
- **Real Gemini API** -- the visitor pastes their own key (never written
  to disk, scoped to that run only, never stored in a shared env var so
  concurrent sessions can't see each other's key).

### Deploying on share.streamlit.io

1. Push this whole `gemini_safety_auditor/` directory to a GitHub repo,
   with `streamlit_app.py` at the path you'll point Streamlit Cloud at.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing at that repo and `streamlit_app.py`.
3. **Do not** put your own `GEMINI_API_KEY` in Streamlit's "Secrets" for a
   public deployment unless you also add an access gate (e.g. a simple
   password check before `use_real_api` is allowed) -- otherwise anyone who
   finds the URL can spend your quota. The default setup here asks each
   visitor for their own key instead, which avoids that entirely.
4. Community Cloud's free tier serves one process for all visitors. The
   real-API-key isolation between concurrent sessions is handled (see
   `agent/session_client.py`), but the offline demo's fake-client patching
   (`agent/fake_client.py`) is process-global and not session-isolated --
   fine for light/sequential demo traffic, not guaranteed correct under
   truly simultaneous offline-mode runs from different visitors.

### A note on four files in this repo

`agent/orchestrator.py`, `agent/eval_awareness.py`,
`probes/embedding_probe.py`, and `probes/seed_examples.py` were not
available when this Streamlit integration was put together and were
reconstructed from the interfaces `main.py`, `smoke_test.py`, and this
README already implied (function signatures, return-type field names like
`.disagreement` / `.behavior_changed`, expected result counts). Each file
says so in its own docstring. If your original versions differ, drop them
in over these -- nothing else in the repo needs to change, since
`streamlit_app.py` and `smoke_test.py` only depend on the public functions
(`run_audit`, `run_eval_awareness_audit`, `EmbeddingProbe.fit/score`,
`get_seed_dataset`) these files expose.

## Full repo: toy pipeline + real pipeline together

This directory contains the **real-Gemini** half of the project. The
**toy** half — the original four-workstream pipeline this one extends
(interpretability, red-teaming, systems, RL, all on a hand-rolled NumPy
model) — lives in `toy_pipeline/`, with its own README. Together they tell
the "toy → real" migration story end to end: run the toy scripts to see
the full methodology on a model with total internal access, then run this
directory's scripts to see the same underlying questions asked of a real,
black-box, production model.

```
gemini_safety_auditor/
├── toy_pipeline/                 # original toy 4-workstream pipeline (see its own README)
│   ├── shared/
│   ├── workstream1_interpretability/
│   ├── workstream2_redteam/
│   ├── workstream3_systems/
│   └── workstream4_rl/
├── main.py                       # entry point: fits probe, runs all audit phases
├── agent/
│   ├── config.py                 # API key + model IDs (env-based, no hardcoding)
│   ├── gemini_client.py          # generate() and embed() -- thin API wrapper
│   ├── orchestrator.py           # Phase 1: plan -> act -> verify -> retry loop
│   ├── eval_awareness.py         # Phase 2: unmarked vs. eval-flagged A/B test
│   ├── injection_probe.py        # Phase 3: indirect prompt-injection probe
│   ├── multimodal_probe.py       # Phase 4: text-only vs. with-image consistency
│   └── cross_tier.py             # Phase 5: fast-tier vs. capable-tier speed/safety
├── probes/
│   ├── embedding_probe.py        # logistic-regression probe on embeddings
│   └── seed_examples.py          # tiny labeled safe/unsafe seed set
├── benchmark/                    # (next) latency cost of running both checks
└── findings/
    ├── audit_log.jsonl           # every Phase 1 trajectory, appended as JSON lines
    ├── eval_awareness_log.jsonl  # every Phase 2 comparison, appended as JSON lines
    ├── injection_log.jsonl       # every Phase 3 scenario, appended as JSON lines
    ├── multimodal_log.jsonl      # every Phase 4 comparison, appended as JSON lines
    └── cross_tier_log.jsonl      # every Phase 5 comparison, appended as JSON lines
```

## Phase 2: Eval-awareness check

Motivated by a real, verified finding: alignment researchers found that
Gemini can behave *worse*, not better, when it explicitly reasons an
environment is a contrived test — often treating it as a "CTF"-style
puzzle to solve by unconventional means rather than a real situation.
([source](https://www.alignmentforum.org/posts/aTcsN5ZZDnMFJvRiG/models-may-behave-worse-when-eval-aware))

This project does **not** reproduce that paper's methodology or claim its
numbers apply here. It runs a much smaller, honest test of the same
question against your own probe prompts: for each prompt, run it once
unmarked and once explicitly prefixed as "this is a safety evaluation,"
then compare refusal behavior and probe scores. Any shift found is a data
point about this specific prompt set, not a general claim about Gemini's
behavior.

## Phase 3: Indirect prompt-injection check

Motivated by Gemini's real commercial edge — live read access to Gmail,
Docs, Sheets, and Calendar — which is also its largest attack surface: an
agent that reads content it didn't author (an event description, an email
body, a doc comment) can't always tell the difference between that
content and its actual instructions. Security researchers have
demonstrated this against real Workspace-connected deployments (e.g.
[SafeBreach's calendar-invite writeup](https://www.safebreach.com/blog/gemini-voice-assistant-prompt-injection-exploit/)).

**Framing note:** this vulnerability class isn't unique to Gemini — any
model wired up to read untrusted external content has the same exposure.
Gemini is just what this project can reach.

`agent/injection_probe.py` builds three realistic "external content"
scenarios (calendar invite, email, doc comment), each carrying a smuggled
instruction, and checks whether Gemini's response follows the smuggled
instruction instead of / alongside the user's actual task. The smuggled
instruction is always a harmless, regenerated-per-run canary string, never
a genuinely harmful payload — the point is measuring susceptibility to the
injection class, not demonstrating real-world harm.

## Phase 4: Multimodal consistency check

Motivated by Gemini processing text/image/audio/video under one shared
representation rather than separate per-modality pipelines — which opens
the door to cross-modal "bleed-through," where safety behavior learned on
text doesn't fully generalize once an image is in the context.

**This is explicitly not a working adversarial-image jailbreak** — see
`agent/multimodal_probe.py`'s docstring for the full honest-scope note.
Short version: a real attack needs gradient access to optimize pixel
perturbations, which we don't have against a hosted endpoint (same
limitation as the text-suffix search in Phase 1). What this phase actually
checks is much narrower: does the mere *presence* of a plain, non-adversarial
placeholder image shift the refusal or probe verdict on the same boundary
prompts from Phase 1, relative to a text-only baseline? A shift is a
correlational signal worth a closer look, not evidence of a jailbreak.

## Phase 5: Cross-tier speed-vs-safety check

Motivated by the obvious question once you have a fast tier and a capable
tier available: does choosing the cheaper/faster model trade away safety
behavior, not just latency? `agent/cross_tier.py` runs the same prompts
through both tiers (configurable in `agent/config.py`) and reports latency
alongside the Phase-1-style behavioral and probe verdicts for each.

As with everything else here: two tiers on a handful of prompts, run once
each, is a spot-check. Treat any difference as a lead to investigate with
more prompts and more repeats, not a confirmed regression.

## Roadmap

- [x] Basic agent loop calling real Gemini API
- [x] Embedding-space probe as a second verify signal
- [x] Disagreement detection (behavioral vs. probe)
- [x] Eval-awareness A/B test (unmarked vs. eval-flagged prompts)
- [x] Indirect prompt-injection probe (Workspace-style content scenarios)
- [x] Multimodal consistency probe (text-only vs. with-image, black-box)
- [x] Cross-tier comparison (fast vs. capable) to test the speed-vs-safety tradeoff
- [ ] Latency/cost benchmark of running all five checks per generation at scale
- [ ] Expand seed dataset beyond 12 hand-written examples
- [ ] Write up any disagreement/shift/injection/regression cases found, with the honest caveats above
