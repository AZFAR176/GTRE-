from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClinicalTextEncoder(nn.Module):
    """Frozen BioClinicalBERT + trainable projection head.

    ``encode_entities`` returns per-entity embeddings z_e in R^d, i.e. one vector
    per clinical entity in the report (used to build the S x V alignment matrix).
    """

    def __init__(self, d_model: int = 256,
                 hf_name: str = "emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer  # lazy import

        self.tokenizer = AutoTokenizer.from_pretrained(hf_name)
        self.bert = AutoModel.from_pretrained(hf_name)
        for p in self.bert.parameters():           # frozen text backbone
            p.requires_grad_(False)
        self.bert.eval()

        hidden = self.bert.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

    @torch.no_grad()
    def _embed_strings(self, strings: Sequence[str], device) -> torch.Tensor:
        tok = self.tokenizer(list(strings), padding=True, truncation=True,
                             max_length=64, return_tensors="pt").to(device)
        out = self.bert(**tok).last_hidden_state            # [n, L, hidden]
        mask = tok["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)  # mean-pool
        return pooled

    def encode_entities(self, entity_lists: List[List[str]]) -> torch.Tensor:
        """entity_lists: B reports, each a list of V entity strings (padded to V).

        Returns z_e: [B, V, d]. The frozen BERT is detached; only ``proj`` trains.
        """
        device = self.proj[0].weight.device
        B = len(entity_lists)
        V = max(len(e) for e in entity_lists)
        flat, lengths = [], []
        for ents in entity_lists:
            ents = ents + [""] * (V - len(ents))
            flat.extend(ents)
            lengths.append(len(ents))
        pooled = self._embed_strings(flat, device)          # [B*V, hidden]
        z_e = self.proj(pooled).view(B, V, -1)              # [B, V, d]
        return F.normalize(z_e, dim=-1)
