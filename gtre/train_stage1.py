"""Stage 1 : Multi-Domain Grounding (contrastive segment-to-token alignment +
domain-invariant prototype regularization).

    L_stage1 = (1/M) sum_m L_seg-clip^m  +  lambda_dom * L_dom

Data is loaded directly from folders (``--signals_dir`` / ``--texts_dir``) of
subject-disjoint EEG-text pairs pooled across domains D_m in {TUSZ, TUAB}
(MIMIC-IV contributes auxiliary text only) via ``EEGTextPairDataset`` +
``DataLoader``. K = 4 global prototypes, lambda_dom = 0.1, Adam lr 5e-4,
batch 1024 (Implementation Details).
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import GTRE, GTREConfig
from .losses import stage1_loss
from .data import (EEGTextPairDataset, collate_pairs, load_montage,
                   load_entity_vocab)


def train_stage1(cfg: GTREConfig, signals_dir: str, texts_dir: str,
                 electrode_pos: torch.Tensor, entity_vocab=None, epochs: int = 20,
                 batch_size: int = 1024, lr: float = 5e-4, lambda_dom: float = 0.1,
                 n_prototypes: int = 4, num_workers: int = 0, device: str = "cpu"):
    from .text_encoder import ClinicalTextEncoder

    model = GTRE(cfg).to(device)
    text_encoder = ClinicalTextEncoder(cfg.d_model).to(device)
    prototypes = nn.Parameter(torch.randn(n_prototypes, cfg.d_model, device=device))

    params = (list(model.parameters())
              + list(text_encoder.proj.parameters())       # only the proj head trains
              + [prototypes])
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: 1.0 if e < int(0.8 * epochs)
        else 0.96 ** (e - int(0.8 * epochs))
    )

    dataset = EEGTextPairDataset(
        signals_dir, texts_dir,
        n_time=cfg.micro_patch_len * cfg.macro_segments * 8,
        n_channels=electrode_pos.size(0), entity_vocab=entity_vocab)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, collate_fn=collate_pairs)
    pos = electrode_pos.to(device)

    for epoch in range(epochs):
        model.train()
        last = None
        for eeg, ents, dom in loader:
            eeg, dom = eeg.to(device), dom.to(device)
            out = model(eeg, pos)                            # segments [B,S,d]
            z_ent = text_encoder.encode_entities(ents)       # [B,V,d]
            stats = stage1_loss(out["segments"], z_ent, dom, prototypes,
                                lambda_dom=lambda_dom)
            opt.zero_grad()
            stats["loss"].backward()
            opt.step()
            last = stats
        sched.step()
        if last is not None:
            print(f"[stage1] epoch {epoch:02d} loss={last['loss'].item():.4f} "
                  f"clip={last['clip'].item():.4f} dom={last['dom'].item():.4f}")

    return {"model": model, "text_encoder": text_encoder,
            "prototypes": prototypes.detach()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals_dir", required=True,
                    help="Folder of EEG signal files ([C,T]); optional per-domain subfolders.")
    ap.add_argument("--texts_dir", required=True,
                    help="Folder of .txt reports (same stem as signal).")
    ap.add_argument("--montage", required=True,
                    help="CSV of electrode coords (name,x,y,z).")
    ap.add_argument("--entity_vocab", help="Curated clinical-entity vocab .txt.")
    ap.add_argument("--out_ckpt", default="stage1_encoder.pt",
                    help="Where to save the trained encoder (feeds --stage1_ckpt).")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = GTREConfig()
    pos, _ = load_montage(args.montage)

    vocab = load_entity_vocab(args.entity_vocab)
    artifacts = train_stage1(cfg, args.signals_dir, args.texts_dir, pos,
                             entity_vocab=vocab, epochs=args.epochs,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers, device=args.device)

    # persist the encoder so Stage 2 can pick it up via --stage1_ckpt
    torch.save(artifacts["model"].state_dict(), args.out_ckpt)
    print(f"[stage1] saved encoder to {args.out_ckpt}")
