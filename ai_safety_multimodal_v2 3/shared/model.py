"""
shared/model.py

A small "multimodal" safety classifier implemented as a real PyTorch
nn.Module, trained with real autograd (torch.optim), instead of hand-derived
NumPy gradients. This is the v2 upgrade over the original toy_multimodal_model.py:
same architecture and same intent (text embedding + image embedding -> fused
hidden layer -> safety score), but now built on the exact framework
(PyTorch) and API shape (forward hooks, .backward()) that every real-model
technique downstream (probes via nnsight/transformer_lens, PGD attacks,
DDPO/PPO) actually expects.

REAL MODEL SWAP
----------------
This sandbox has no access to huggingface.co (only pypi/github/npm are
reachable), so we can't download real VLM weights here. But the class below
is written so the swap is a single method: MultimodalSafetyModel.forward()
takes a batch of (text_emb, image_emb) tensors and returns (logits, hidden).
To point this at a real model on a machine with GPU + internet:

    from transformers import AutoModel, AutoProcessor
    vlm = AutoModel.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    def get_activations(text, image):
        inputs = processor(text=text, images=image, return_tensors="pt")
        with torch.no_grad():
            out = vlm(**inputs, output_hidden_states=True)
        return out.hidden_states[-1]  # (batch, seq, hidden_dim) residual stream

    # then register a forward hook at whichever layer you want to probe/patch,
    # e.g. vlm.model.layers[12].register_forward_hook(...)

Everything downstream (probe_and_patch.py, attack_loop.py, benchmark.py,
rlhf_loop.py) only depends on the (logits, hidden) contract below, so once
you have `get_activations`, those scripts need only their model-construction
lines changed -- the probe/patch/attack/RL logic is untouched.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalSafetyModel(nn.Module):
    """
    Input -> Linear -> ReLU (hidden = "residual stream") -> Linear -> logit.
    Structurally identical to the v1 NumPy model, but now a real nn.Module
    so gradients, hooks, and optimizers all work the way they would on a
    real transformer.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, seed: int = 1):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, patch_fn=None, return_hidden: bool = False):
        """
        x: (N, input_dim) float tensor
        patch_fn: optional callable(hidden) -> hidden, applied to the hidden
                  layer before the readout (used for activation patching /
                  steering). Same contract as v1.
        """
        h = F.relu(self.fc1(x))
        if patch_fn is not None:
            h = patch_fn(h)
        logit = self.fc2(h).squeeze(-1)
        if return_hidden:
            return logit, h
        return logit

    def predict_proba(self, x: torch.Tensor, patch_fn=None) -> torch.Tensor:
        with torch.no_grad():
            logit = self.forward(x, patch_fn=patch_fn)
        return torch.sigmoid(logit)

    def fit(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 300, lr: float = 0.05,
            weight_decay: float = 1e-4, verbose: bool = False):
        """Real training loop: Adam + BCEWithLogitsLoss, not hand-derived gradients."""
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        for epoch in range(epochs):
            opt.zero_grad()
            logits = self.forward(X)
            loss = loss_fn(logits, y.float())
            loss.backward()
            opt.step()
            if verbose and epoch % 50 == 0:
                with torch.no_grad():
                    acc = ((torch.sigmoid(logits) > 0.5).long() == y).float().mean().item()
                print(f"  epoch {epoch:4d}  loss={loss.item():.4f}  acc={acc:.3f}")
        return self


def register_activation_hook(model: MultimodalSafetyModel, patch_fn):
    """
    Real forward-hook based steering, the way you'd do it on an actual
    transformer layer (`layer.register_forward_hook(...)`). Returns a
    handle; call handle.remove() to undo. This is an alternative to passing
    patch_fn directly into forward() -- useful once you're hooking into a
    real HF model where you don't control the forward() signature.
    """
    def hook(module, inp, out):
        return patch_fn(out)
    return model.fc1.register_forward_hook(hook)
