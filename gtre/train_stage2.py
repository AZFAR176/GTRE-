"""Stage 2 : Fine-Grained Preference Refinement via Masked-DPO.

Initialized from the grounded Stage-1 encoder pi_theta; a frozen *copy* serves as
the reference pi_ref. For each preference triplet (X_eeg, t^+, t^-) with a
pathology-aware mask M (loaded from folders via ``PreferenceDataset``):

    s_theta^+- = cos(z_e^theta, M (x) z_{t^+-})     (trained encoder)
    s_ref^+-   = cos(z_e^ref,   M (x) z_{t^+-})     (frozen reference)
    delta      = cos(z_{t^+}, z_{t^-})

    L_total = L_m-dpo (Eq. 3)  +  lambda_proto * L_proto.

beta = 0.1, adaptive margin lambda = 0.3, K = 8 semantic prototypes.
"""

from __future__ import annotations

import argparse
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import GTRE, GTREConfig
from .losses import masked_similarity, masked_dpo_loss, consistency_proto_loss
from .data import (PreferenceDataset, collate_preferences, pad_mask,
                   load_montage, load_entity_vocab)


# ---------------------------------------------------------------------------
# Reward-proxy distillation (Automated Preference Distillation) -- offline stub
# ---------------------------------------------------------------------------
class RewardProxy(nn.Module):
    """Lightweight clinical reward proxy trained on doctor-validated triplets;
    at scale it ranks GPT-generated summaries (prefers verified clinical entities
    over generic phrasing) to build the (t^+, t^-) folders consumed here."""

    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(),
                                   nn.Linear(d_model // 2, 1))

    def forward(self, z_text: torch.Tensor) -> torch.Tensor:
        return self.score(z_text).squeeze(-1)               # higher = preferred


def _masked_pool(z_entities: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1)
    return (z_entities * m).sum(1) / m.sum(1).clamp(min=1e-8)


def train_stage2(stage1_artifacts: dict, cfg: GTREConfig, signals_dir: str,
                 pos_dir: str, neg_dir: str, electrode_pos: torch.Tensor,
                 entity_vocab=None, epochs: int = 20, batch_size: int = 1024,
                 lr: float = 5e-4, beta: float = 0.1, lam_margin: float = 0.3,
                 lam_proto: float = 0.3, n_prototypes: int = 8,
                 num_workers: int = 0, device: str = "cpu"):
    model: GTRE = stage1_artifacts["model"].to(device)      # pi_theta (trainable)
    text_encoder = stage1_artifacts["text_encoder"].to(device)

    ref_model = copy.deepcopy(model).to(device)             # pi_ref (frozen)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()

    prototypes = nn.Parameter(torch.randn(n_prototypes, cfg.d_model, device=device))
    params = list(model.parameters()) + list(text_encoder.proj.parameters()) + [prototypes]
    opt = torch.optim.Adam(params, lr=lr)

    dataset = PreferenceDataset(
        signals_dir, pos_dir, neg_dir,
        n_time=cfg.micro_patch_len * cfg.macro_segments * 8,
        n_channels=electrode_pos.size(0), entity_vocab=entity_vocab)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, collate_fn=collate_preferences)
    pos = electrode_pos.to(device)

    for epoch in range(epochs):
        model.train()
        last = None
        for eeg, tpos, tneg, mpos, mneg in loader:
            eeg = eeg.to(device)
            z_e = model(eeg, pos)["z_eeg"]                  # trained encoder
            with torch.no_grad():
                z_e_ref = ref_model(eeg, pos)["z_eeg"]      # frozen reference

            z_tpos = text_encoder.encode_entities(tpos)     # [B, V+, d]
            z_tneg = text_encoder.encode_entities(tneg)     # [B, V-, d]
            Mpos = pad_mask(mpos, z_tpos.size(1), device)
            Mneg = pad_mask(mneg, z_tneg.size(1), device)

            s_theta_pos = masked_similarity(z_e, z_tpos, Mpos)
            s_theta_neg = masked_similarity(z_e, z_tneg, Mneg)
            s_ref_pos = masked_similarity(z_e_ref, z_tpos, Mpos)
            s_ref_neg = masked_similarity(z_e_ref, z_tneg, Mneg)

            # delta = cos(z_{t+}, z_{t-}) on masked (pooled) text embeddings
            delta = torch.cosine_similarity(_masked_pool(z_tpos, Mpos),
                                            _masked_pool(z_tneg, Mneg), dim=-1)

            l_dpo = masked_dpo_loss(s_theta_pos, s_theta_neg, s_ref_pos,
                                    s_ref_neg, delta, beta=beta, lam=lam_margin)
            l_proto = consistency_proto_loss(z_e_ref, z_e, prototypes)
            loss = l_dpo + lam_proto * l_proto

            opt.zero_grad()
            loss.backward()
            opt.step()
            last = (loss, l_dpo, l_proto)
        if last is not None:
            print(f"[stage2] epoch {epoch:02d} loss={last[0].item():.4f} "
                  f"dpo={last[1].item():.4f} proto={last[2].item():.4f}")

    return {"model": model, "text_encoder": text_encoder,
            "prototypes": prototypes.detach()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals_dir", required=True,
                    help="Folder of EEG signal files ([C,T]).")
    ap.add_argument("--pos_dir", required=True,
                    help="Folder of preferred summaries t^+ (.txt).")
    ap.add_argument("--neg_dir", required=True,
                    help="Folder of dispreferred summaries t^- (.txt).")
    ap.add_argument("--montage", required=True,
                    help="CSV of electrode coords (name,x,y,z).")
    ap.add_argument("--entity_vocab", help="Curated clinical-entity vocab .txt.")
    ap.add_argument("--stage1_ckpt", help="Optional Stage-1 encoder checkpoint (.pt).")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = GTREConfig()
    pos, _ = load_montage(args.montage)
    vocab = load_entity_vocab(args.entity_vocab)

    # load (or freshly build) the Stage-1 encoder to initialize pi_theta
    from .text_encoder import ClinicalTextEncoder
    model = GTRE(cfg)
    if args.stage1_ckpt:
        model.load_state_dict(torch.load(args.stage1_ckpt, map_location="cpu"))
        print(f"[stage2] loaded Stage-1 encoder from {args.stage1_ckpt}")
    else:
        print("[stage2] WARNING: no --stage1_ckpt; pi_theta starts from scratch.")
    artifacts = {"model": model, "text_encoder": ClinicalTextEncoder(cfg.d_model)}

    train_stage2(artifacts, cfg, args.signals_dir, args.pos_dir, args.neg_dir, pos,
                 entity_vocab=vocab, epochs=args.epochs, batch_size=args.batch_size,
                 num_workers=args.num_workers, device=args.device)
