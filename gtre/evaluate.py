from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Single-layer MLP classification head
# ---------------------------------------------------------------------------
class MLPHead(nn.Module):
    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_model, n_classes)             # single linear layer

    def forward(self, z):
        return self.fc(z)


def _encode(model, eeg, pos):
    return model(eeg, pos)["z_eeg"]


@torch.no_grad()
def _accuracy(model, head, loader, pos, device):
    model.eval(); head.eval()
    correct = total = 0
    for eeg, y in loader:
        eeg, y = eeg.to(device), y.to(device)
        logits = head(_encode(model, eeg, pos))
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------
def linear_probe(model, train_loader, test_loader, pos, n_classes,
                 epochs=20, lr=5e-4, device="cpu"):
    """Encoder frozen; train only the single-layer head."""
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    head = MLPHead(model.cfg.d_model, n_classes).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    pos = pos.to(device)
    for _ in range(epochs):
        head.train()
        for eeg, y in train_loader:
            eeg, y = eeg.to(device), y.to(device)
            with torch.no_grad():
                z = _encode(model, eeg, pos)
            loss = F.cross_entropy(head(z), y)
            opt.zero_grad(); loss.backward(); opt.step()
    return _accuracy(model, head, test_loader, pos, device)


def few_shot(model, fewshot_loader, test_loader, pos, n_classes,
             epochs=20, lr=5e-4, device="cpu"):
    """Few-shot (5% labels): fine-tune encoder + head jointly."""
    model = model.to(device).train()
    for p in model.parameters():
        p.requires_grad_(True)
    head = MLPHead(model.cfg.d_model, n_classes).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=lr)
    pos = pos.to(device)
    for _ in range(epochs):
        for eeg, y in fewshot_loader:                       # 5% labeled subset
            eeg, y = eeg.to(device), y.to(device)
            loss = F.cross_entropy(head(_encode(model, eeg, pos)), y)
            opt.zero_grad(); loss.backward(); opt.step()
    return _accuracy(model, head, test_loader, pos, device)


def full_finetune(model, train_loader, test_loader, pos, n_classes,
                  epochs=20, lr=5e-4, device="cpu"):
    """SOTA setting: full fine-tuning on all labels."""
    return few_shot(model, train_loader, test_loader, pos, n_classes,
                    epochs=epochs, lr=lr, device=device)


@torch.no_grad()
def zero_shot(model, text_encoder, test_loader, pos, class_entities: List[List[str]],
              device="cpu"):
    """Assign each recording to the class whose medical-entity embedding is the
    most cosine-similar to z_eeg (no task-specific training)."""
    model = model.to(device).eval()
    pos = pos.to(device)
    # one prototype embedding per class (mean over its entity strings)
    z_classes = text_encoder.encode_entities(class_entities).mean(dim=1)  # [n_cls, d]
    z_classes = F.normalize(z_classes, dim=-1)
    correct = total = 0
    for eeg, y in test_loader:
        eeg, y = eeg.to(device), y.to(device)
        z = F.normalize(_encode(model, eeg, pos), dim=-1)   # [B, d]
        sims = z @ z_classes.t()                            # cosine similarity
        correct += (sims.argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)
