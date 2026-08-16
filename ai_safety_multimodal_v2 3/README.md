# AI Safety & Alignment for Multimodal Generative AI (v2)

A runnable demonstration of four AI safety workstreams — **Mechanistic
Interpretability**, **AI Security / Red Teaming**, **ML Systems &
Performance**, and **RL Alignment** — applied to one shared toy multimodal
model, built so the four techniques compose into a single cross-cutting
finding: *can interpretability tools catch reward hacking that a safety
metric alone would miss?*

v2 replaces every hand-rolled NumPy component with the real library/
algorithm you'd use on a real model (see table below), adds a
generalization test that the v1 version didn't have, and adds automated
tests, plots, and a results report.

## What changed from v1 → v2

| Component | v1 | v2 |
|---|---|---|
| Model + training | Hand-derived NumPy gradients | Real `torch.nn.Module` + `torch.optim.Adam` |
| Activation patching | Function argument | Real `register_forward_hook` |
| Red-team attack | Gradient-free hill-climbing | Real **PGD** (`torch.autograd`), plus an ASR-vs-budget sweep |
| SAE | Hand-derived gradients | Real `nn.Module` + Adam + dead-feature tracking |
| RL | Hand-derived REINFORCE | `torch.distributions.Normal` + `torch.optim.Adam` |
| Dataset | 8 concepts, one split | 24 concepts, **train/held-out concept split** for a genuine generalization test |
| Evidence | Terminal logs only | `results/*.png` plots + `results/summary.json` + auto-generated `results/REPORT.md` |
| Tests | None | `pytest` smoke tests (`tests/test_pipeline.py`) |

## Why still a toy model

This sandbox (and the one it was originally built in) has **no access to
huggingface.co** — only PyPI/GitHub are reachable — so real VLM weights
can't be downloaded here. `shared/model.py` is a real PyTorch `nn.Module`
with real autograd; the only thing still "toy" is that it's trained from
scratch on synthetic embeddings rather than being a pretrained transformer.
Every *technique* applied to it (PGD, forward hooks, SAEs, REINFORCE) is the
real technique — see `shared/model.py`'s docstring for the exact code path
to swap in a real HF VLM once you have GPU + internet access.

## An honest limitation worth knowing about (and mentioning if asked)

The held-out-concept generalization test in Workstream 1 shows the probe
direction does **not** generalize well (42.5% accuracy on concepts unseen
in training, vs. 100% on training concepts). This isn't a bug — each
concept (`"a bomb"`, `"a bioweapon"`, etc.) gets its own independent random
embedding vector with no shared semantic structure, so there is no
geometric relationship for a linear probe to generalize along. On a real
VLM, concept embeddings *do* share structure (that's what makes them
useful), so this exact failure mode wouldn't necessarily reproduce — but
it's a legitimate thing to flag if someone asks how far this toy setup's
findings should be trusted to transfer.

## Repository structure

```
ai_safety_multimodal_v2/
├── requirements.txt
├── run_all.py                      # runs all 4 workstreams + writes results/REPORT.md
├── shared/
│   ├── model.py                    # real PyTorch multimodal model + real-VLM swap instructions
│   ├── dataset.py                  # 24-concept synthetic dataset + train/held-out concept split
│   ├── eval_harness.py             # accuracy / safety violation rate / false positive rate
│   └── utils.py                    # results/plot saving helpers
├── workstream1_interpretability/probe_and_patch.py   # probe -> hook-based patch -> SAE -> generalization test
├── workstream2_redteam/attack_loop.py                # real PGD attack -> ASR sweep -> adversarial fine-tune
├── workstream3_systems/benchmark.py                  # latency/throughput of guardrails, plotted
├── workstream4_rl/rlhf_loop.py                       # REINFORCE alignment + WS1 reward-hacking audit
├── tests/test_pipeline.py          # pytest smoke tests
└── results/                        # generated: summary.json, REPORT.md, *.png
```

## Running it

```bash
pip install -r requirements.txt

python3 run_all.py          # runs everything, writes results/REPORT.md + plots
pytest tests/ -v             # sanity-check the pipeline
```

Or run any workstream individually, e.g. `python3 workstream2_redteam/attack_loop.py`.

## The cross-cutting finding

1. **Workstream 1** finds a linear direction in the hidden layer that's
   causally load-bearing for safety behavior (ablating it via a real
   forward hook collapses the model into over-flagging), but that
   direction **does not generalize** to unseen concepts in this toy setup
   — a genuine, reportable limitation.
2. **Workstream 2** runs a real PGD attack and shows the model is
   increasingly exploitable as the perturbation budget grows (0% ASR at
   budget 0.25 → 52.5% at budget 3.0), and that adversarial fine-tuning
   trades attack success rate (51.3% → 6.7%) for a large false-positive
   cost (96.0%) — the standard robustness/usability tradeoff.
3. **Workstream 3** shows the resulting guardrail stack (classifier +
   steering hook) costs +137% latency vs. the base model at batch=256 —
   the number an infra team needs to decide on an early-exit strategy.
4. **Workstream 4** runs real REINFORCE alignment against the Workstream 1
   reward model, then re-uses the Workstream 1 probe as an auditor: in
   this run, the unsafe-direction projection **decreased** alongside the
   violation rate (2.626 → -8.484), indicating genuine suppression rather
   than reward hacking — but the script is written to flag the opposite
   case explicitly if it occurs (it's a real check, not a scripted
   outcome — flip the random seed and it can go either way).

Full numbers: [`results/REPORT.md`](results/REPORT.md) after running `run_all.py`.

## Real-model case study: sycophancy DPO audit

`real_model_case_study/sycophancy_dpo_audit/` is the real-model version of
this same pattern — not a hypothetical, an actual pipeline: Gemma-3-4B +
a Gemma Scope SAE + DPO fine-tuning, auditing whether preference-tuning a
model out of sycophancy (caving to a user's confidently wrong claim)
produces genuine suppression or just a fix that looks clean until you
red-team it with pressure framings it never saw in training. Same
four-workstream shape as the toy pipeline above (alignment /
interpretability / red-team / systems), same cross-cutting question (did
the fix generalize, or did it reward-hack the metric).

It ships with a Streamlit dashboard (`streamlit_app.py`) that's
independent of the GPU training code — deployable to streamlit.io for
free — plus a Colab-ready pipeline for the actual training. See that
folder's own README for exact run/deploy steps.

## Path to a real model (needs GPU + internet)

```python
from transformers import AutoModel, AutoProcessor
vlm = AutoModel.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

def get_activations(text, image):
    inputs = processor(text=text, images=image, return_tensors="pt")
    with torch.no_grad():
        out = vlm(**inputs, output_hidden_states=True)
    return out.hidden_states[-1]  # real residual stream
```

Everything downstream (probe, hook-based patch, PGD attack, benchmark, RL
loop) depends only on the `(logits, hidden)` contract in `model.py`'s
`forward()`, so this swap changes model-construction lines only — the
probe/patch/attack/RL logic in each workstream script is untouched. See
`shared/model.py`'s docstring for the equivalent for a diffusion model +
DDPO, and `shared/dataset.py`'s `load_real_dataset()` for pulling in a real
jailbreak/safety benchmark dataset (e.g. AdvBench, JailbreakBench) once you
have that network access.
