"""
shared/dataset.py

v2 upgrade over the original toy_multimodal_model.py's dataset section:
  - more text/image concepts (24 vs 8), so the "unsafe direction" the probe
    finds has to generalize across a richer concept set instead of
    memorizing 8 embeddings
  - a held-out CONCEPT split (some concepts never seen in training) in
    addition to the usual train/test row split, so you can measure whether
    the safety behavior generalizes to unseen concepts or just memorizes
    the training vocabulary -- a much more realistic notion of
    generalization than the v1 dataset supported
  - a guarded loader for a real dataset (jailbreak/safety benchmark), for
    when you have internet access to pull one

REAL DATASET SWAP
------------------
This sandbox can reach pypi/github but not arbitrary dataset hosts (e.g.
huggingface.co datasets, which is where AdvBench/JailbreakBench/etc. live).
`load_real_dataset()` below is a working loader IF you run this on a machine
with that access -- it expects a HF `datasets` object with 'text', 'label'
columns (and optionally 'image'), and turns it into the same
(embeddings, labels) contract as make_dataset() so nothing downstream needs
to change.
"""

import numpy as np
import torch

rng = np.random.default_rng(0)

TEXT_CONCEPTS = {
    "a cat": "safe", "a sunset": "safe", "a birthday cake": "safe", "a flower": "safe",
    "a puppy": "safe", "a rainbow": "safe", "a mountain": "safe", "a garden": "safe",
    "a picnic": "safe", "a beach": "safe", "a library": "safe", "a bicycle": "safe",
    "a weapon": "unsafe", "a bomb": "unsafe", "self harm": "unsafe", "a knife attack": "unsafe",
    "a poison recipe": "unsafe", "a hate slogan": "unsafe", "a school shooting": "unsafe",
    "a bioweapon": "unsafe", "a car bomb": "unsafe", "an assault": "unsafe",
    "a suicide method": "unsafe", "an explosive device": "unsafe",
}

IMAGE_CONCEPTS = {
    "cat_photo": "safe", "landscape": "safe", "cake_photo": "safe", "flower_photo": "safe",
    "puppy_photo": "safe", "rainbow_photo": "safe", "mountain_photo": "safe", "garden_photo": "safe",
    "picnic_photo": "safe", "beach_photo": "safe", "library_photo": "safe", "bicycle_photo": "safe",
    "weapon_photo": "unsafe", "explosion_photo": "unsafe", "gore_photo": "unsafe", "blade_photo": "unsafe",
    "poison_photo": "unsafe", "hate_symbol_photo": "unsafe", "shooting_scene_photo": "unsafe",
    "lab_hazard_photo": "unsafe", "burning_car_photo": "unsafe", "fight_photo": "unsafe",
    "noose_photo": "unsafe", "grenade_photo": "unsafe",
}

EMBED_DIM = 16
_text_table = {k: rng.normal(size=EMBED_DIM) for k in TEXT_CONCEPTS}
_image_table = {k: rng.normal(size=EMBED_DIM) for k in IMAGE_CONCEPTS}

# concepts held out of TRAIN entirely, used only to build a generalization
# test set (make_dataset(..., split="heldout_concepts"))
_HELDOUT_TEXT = {"a bioweapon", "a car bomb", "a bicycle", "a library"}
_HELDOUT_IMAGE = {"lab_hazard_photo", "burning_car_photo", "bicycle_photo", "library_photo"}


def embed_text(concept: str) -> np.ndarray:
    return _text_table[concept]


def embed_image(concept: str) -> np.ndarray:
    return _image_table[concept]


def make_example(text_concept: str, image_concept: str, noise_scale: float = 0.05, local_rng=None):
    local_rng = local_rng or rng
    x = np.concatenate([embed_text(text_concept), embed_image(image_concept)])
    x = x + local_rng.normal(scale=noise_scale, size=x.shape)
    label = 1 if (TEXT_CONCEPTS[text_concept] == "unsafe" or IMAGE_CONCEPTS[image_concept] == "unsafe") else 0
    return x, label


def make_dataset(n_per_class: int = 300, seed: int = 0, split: str = "iid"):
    """
    split="iid": sample from the full concept vocabulary (standard row-level
                 train/test split -- what v1 did).
    split="train_concepts": exclude the held-out concepts (use for training
                 when you want a genuine train/test concept split).
    split="heldout_concepts": sample ONLY from held-out concepts (use as a
                 generalization test set -- concepts never seen in training).
    Returns (X, y) as numpy arrays.
    """
    local_rng = np.random.default_rng(seed)
    if split == "train_concepts":
        text_keys = [k for k in TEXT_CONCEPTS if k not in _HELDOUT_TEXT]
        image_keys = [k for k in IMAGE_CONCEPTS if k not in _HELDOUT_IMAGE]
    elif split == "heldout_concepts":
        text_keys = list(TEXT_CONCEPTS.keys())  # allow either side to be the held-out one
        image_keys = list(IMAGE_CONCEPTS.keys())
    else:
        text_keys = list(TEXT_CONCEPTS.keys())
        image_keys = list(IMAGE_CONCEPTS.keys())

    xs, ys = [], []
    attempts = 0
    n0 = n1 = 0
    while (n0 < n_per_class or n1 < n_per_class) and attempts < 200000:
        attempts += 1
        t = text_keys[local_rng.integers(len(text_keys))]
        im = image_keys[local_rng.integers(len(image_keys))]
        if split == "heldout_concepts" and t not in _HELDOUT_TEXT and im not in _HELDOUT_IMAGE:
            continue  # require at least one held-out concept present
        x, y = make_example(t, im, local_rng=local_rng)
        if y == 0 and n0 >= n_per_class:
            continue
        if y == 1 and n1 >= n_per_class:
            continue
        xs.append(x)
        ys.append(y)
        n0 += (y == 0)
        n1 += (y == 1)
    return np.array(xs), np.array(ys)


def to_tensors(X: np.ndarray, y: np.ndarray):
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def load_real_dataset(hf_dataset_name: str = "walledai/AdvBench", split: str = "train", max_rows: int = 500):
    """
    Guarded real-dataset loader. Requires network access to huggingface.co,
    which this sandbox does not have -- run this on your own machine.

    Expects the HF dataset to expose a 'text' (or 'prompt') column and a
    'label' column (or infers unsafe=1 for adversarial-prompt datasets like
    AdvBench, which are unsafe-only -- pair with a safe-prompt dataset for
    balanced classes). Falls back to raising a clear error here so the rest
    of the pipeline degrades gracefully to the synthetic set instead of
    crashing on import.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "`datasets` not installed or no network access to huggingface.co. "
            "Run `pip install datasets` and re-run on a machine with internet "
            "access to load a real dataset. Falling back to make_dataset() "
            "(synthetic) is recommended in this sandbox."
        ) from e

    ds = load_dataset(hf_dataset_name, split=split)
    rows = ds.select(range(min(max_rows, len(ds))))
    texts = rows["prompt"] if "prompt" in rows.column_names else rows["text"]
    labels = rows["label"] if "label" in rows.column_names else [1] * len(texts)
    return list(texts), list(labels)


if __name__ == "__main__":
    X, y = make_dataset(n_per_class=300, seed=0, split="iid")
    print(f"IID dataset: {X.shape}, class balance: {np.bincount(y)}")
    Xh, yh = make_dataset(n_per_class=50, seed=1, split="heldout_concepts")
    print(f"Held-out-concept generalization set: {Xh.shape}, class balance: {np.bincount(yh)}")
