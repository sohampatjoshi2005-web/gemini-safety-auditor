# Sycophancy DPO Audit — Real-Model Case Study

This is the **real-model counterpart** to the toy `ai_safety_multimodal_v2`
pipeline one level up: the same pattern (alignment → interpretability →
red-team → systems, with a cross-cutting reward-hacking check) but run on
a real model — Gemma-3-4B + a Gemma Scope SAE + DPO — instead of a toy
NumPy/PyTorch classifier.

## Why this needs two different places to run

| Piece | Needs | Where it runs |
|---|---|---|
| `pipeline.py` (Steps 1–9: load model, DPO train, probe, attack, benchmark) | GPU, ~8GB+ VRAM, gated HF model access | **Colab** (T4 free tier is enough) or your own GPU machine |
| `streamlit_app.py` (dashboard) | Just `streamlit` + `pandas` | **Anywhere** — local machine, VS Code terminal, or streamlit.io |

The dashboard has **zero dependency on torch/transformers/trl** — it only
reads `results/summary.json`. That split is what makes it possible to host
the dashboard for free on streamlit.io, which has no GPU and can't run the
actual training.

## 1. Run the real pipeline in Colab

This is the original notebook (`sycophancy_dpo_audit__3_.ipynb`), refactored
into `pipeline.py` so it also saves results. In a Colab cell:

```python
!pip install -q -U transformers accelerate peft trl sae-lens bitsandbytes huggingface_hub datasets

# paste pipeline.py's contents into a cell, or:
# !wget <raw-github-url-to-pipeline.py> -O pipeline.py

from pipeline import run_all
run_all(hf_token="hf_...")   # needs a token with access to the gated Gemma repos
```

Before running:
1. Accept the license on `google/gemma-3-4b-it` and `google/gemma-scope-2-4b-it` on huggingface.co
2. Create an HF access token (Settings → Access Tokens, read access)
3. Runtime → Change runtime type → T4 GPU
4. **Verify `SAE_RELEASE` / `SAE_ID` in `pipeline.py`** against the current Gemma Scope HF page before running — these version strings change over time
5. Expand `SEED_FACTS` from 3 to 30-50+ before treating any number as a real result rather than a smoke test (flagged in the code)

This writes `results/summary.json` inside the Colab runtime. Download that
`results/` folder (or the whole `sycophancy_dpo_audit/` directory if you
mounted Drive) back to this project folder on your machine, replacing the
empty `results/` folder here.

## 2. Test the dashboard locally in a VS Code terminal

```bash
cd real_model_case_study/sycophancy_dpo_audit
pip install -r requirements_streamlit.txt
streamlit run streamlit_app.py
```

This opens `http://localhost:8501`. If `results/summary.json` isn't there
yet, the app shows clearly-labeled **sample/demo data** so you can check
the layout before you have real numbers — verified working in this repo's
CI-style check (the app starts cleanly and serves HTTP 200 with no
results file present). Once you've copied real results from Colab into
`results/summary.json`, refresh the page — it reads the file on each load,
no server restart needed.

## 3. Host on streamlit.io

1. Push this repo to GitHub (make sure `results/summary.json` — real or
   demo — is committed, since Streamlit Cloud has no Colab session to pull
   it from)
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo
3. **Main file path:** `real_model_case_study/sycophancy_dpo_audit/streamlit_app.py`
4. **Requirements file:** point it at `requirements_streamlit.txt`, *not*
   `requirements_colab.txt` — the free tier will fail trying to install
   `bitsandbytes`/`torch` and has no GPU to run them on anyway
5. Deploy. The public dashboard will show whatever `results/summary.json`
   you committed — re-run the Colab pipeline periodically and push updated
   results to refresh it, rather than trying to run training from the
   hosted app itself.

## What's real vs. what's a placeholder

- `pipeline.py` is a structural refactor of the original notebook (functions
  + result-saving), not a rewrite of the methodology — same DPO/SAE/PGD-style
  logic throughout.
- `streamlit_app.py` was actually run and verified in this environment
  (starts cleanly, serves HTTP 200, renders the demo-data fallback path
  without exceptions) — the demo numbers in `DEMO_DATA` are made up for
  layout-checking purposes only and are labeled as such in the UI; they are
  **not** a real Gemma run.
- The actual Gemma-3-4B + DPO + SAE training was **not** run anywhere in
  building this — that requires GPU + gated HF access this sandbox doesn't
  have. Running it is on you, in Colab, per the steps above.
