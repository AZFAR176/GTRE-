from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class GTREConfig:
    d_model: int = 256          # latent dim d (Z_macro in R^256)
    n_layers: int = 8           # L
    n_heads: int = 8            # H
    ffn_dim: int = 800          # FFN inner dim
    dropout: float = 0.1
    micro_patch_len: int = 50   # 200 ms @ 250 Hz = 50 samples
    micro_patch_stride: int = 25  # overlapping micro-patches
    macro_segments: int = 5     # S : number of 5 s macro-segments per record
    spatial_hidden: int = 128   # hidden width of the 3-layer geometric MLP


# ---------------------------------------------------------------------------
# Topographical coordinate embedding  e^pos = MLP_geo(x, y, z)  in R^d
# ---------------------------------------------------------------------------
class TopographicalCoordinateEmbedding(nn.Module):
    """3-layer MLP mapping 3D electrode coordinates (10-20 system) to R^d.

    The embedding is *added* to the per-channel temporal features so that each
    electrode becomes a spatially anchored token (montage-agnostic).
    """

    def __init__(self, d_model: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        # pos: [C, 3] (or [B, C, 3]) -> [C, d] (or [B, C, d])
        return self.mlp(pos)


# ---------------------------------------------------------------------------
# Temporal hierarchical encoding:  Z_micro (200 ms) -> Z_macro (5 s)
# ---------------------------------------------------------------------------
class _IntraPatchPositionalEncoding(nn.Module):
    """Sinusoidal PE applied *inside* each micro-patch so a token does not
    collapse all temporal information (preserves spikes / morphology)."""

    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [*, L, dim]
        return x + self.pe[:, : x.size(-2)]


class TemporalHierarchicalEncoder(nn.Module):
    """Two-scale temporal encoder.

    Tier 1: raw signal -> overlapping 200 ms micro-patches (Z_micro), each
            embedded with intra-patch positional encoding.
    Tier 2: micro tokens are grouped into fixed, non-overlapping 5 s windows and
            aggregated by self-attention into macro-segments (Z_macro in R^d).

    Returns per-channel macro-segment embeddings: [B, C, S, d].
    """

    def __init__(self, cfg: GTREConfig):
        super().__init__()
        self.cfg = cfg
        # micro-patch embedding (per-sample linear projection of a patch)
        self.micro_proj = nn.Linear(cfg.micro_patch_len, cfg.d_model)
        self.intra_pe = _IntraPatchPositionalEncoding(cfg.d_model)
        # intra-segment aggregation of micro tokens -> one macro embedding
        agg_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout, batch_first=True, activation="gelu",
        )
        self.macro_aggregator = nn.TransformerEncoder(agg_layer, num_layers=2)
        self.macro_query = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)

    def _to_micro_patches(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> micro patches [B, C, P, patch_len]
        B, C, T = x.shape
        patches = x.unfold(dimension=-1, size=self.cfg.micro_patch_len,
                           step=self.cfg.micro_patch_stride)  # [B, C, P, patch_len]
        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B, C, T = x.shape
        patches = self._to_micro_patches(x)            # [B, C, P, patch_len]
        P = patches.size(2)
        z_micro = self.micro_proj(patches)             # [B, C, P, d]  (Z_micro)
        z_micro = self.intra_pe(z_micro)

        # split P micro tokens into S non-overlapping macro windows
        S = cfg.macro_segments
        usable = (P // S) * S
        z_micro = z_micro[:, :, :usable, :]
        win = usable // S
        z_micro = z_micro.reshape(B, C, S, win, cfg.d_model)

        # attention-pool each window into one macro-segment embedding
        z = z_micro.reshape(B * C * S, win, cfg.d_model)
        q = self.macro_query.expand(z.size(0), -1, -1)
        tokens = torch.cat([q, z], dim=1)              # prepend a learnable query
        agg = self.macro_aggregator(tokens)
        z_macro = agg[:, 0, :].reshape(B, C, S, cfg.d_model)  # [B, C, S, d]
        return z_macro


# ---------------------------------------------------------------------------
# Graph-Aware Self-Attention  (Eq. 1):
#   Attn(i,j) = softmax( Q_i K_j^T / sqrt(d_k) - gamma * dist(pos_i, pos_j) )
# ---------------------------------------------------------------------------
class GraphAwareAttention(nn.Module):
    """Multi-head attention over electrode tokens with an additive geometric
    bias. ``gamma`` is a learnable scalar; the full content score Q K^T is
    preserved (soft bias, not a hard mask) so distant electrodes still attend.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # learnable scalar gamma; softplus keeps it a non-negative distance
        # penalty whose *magnitude* is learned (>=0 damps distant electrodes).
        self.gamma_raw = nn.Parameter(torch.tensor(0.0))

    @property
    def gamma(self) -> torch.Tensor:
        return F.softplus(self.gamma_raw)

    def forward(self, x: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        # x: [B, N, d]  ;  dist: [N, N] pairwise electrode distances
        B, N, _ = x.shape
        H, d_k = self.n_heads, self.d_k

        def split(t):
            return t.view(B, N, H, d_k).transpose(1, 2)  # [B, H, N, d_k]

        q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)   # [B, H, N, N]
        scores = scores - self.gamma * dist.unsqueeze(0).unsqueeze(0)  # geometric bias
        attn = self.dropout(scores.softmax(dim=-1))
        out = attn @ v                                        # [B, H, N, d_k]
        out = out.transpose(1, 2).contiguous().view(B, N, H * d_k)
        return self.out_proj(out)


class GraphTransformerLayer(nn.Module):
    """One encoder layer: Graph-Aware MHA + FFN, each wrapped by a residual
    "Add & LayerNorm" sublayer (post-norm, as drawn in Fig. 1)."""

    def __init__(self, cfg: GTREConfig):
        super().__init__()
        self.attn = GraphAwareAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn_dim), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.ffn_dim, cfg.d_model),
        )
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attn(x, dist)))   # Add & LayerNorm
        x = self.norm2(x + self.dropout(self.ffn(x)))          # Add & LayerNorm
        return x


# ---------------------------------------------------------------------------
# Full GTRE model
# ---------------------------------------------------------------------------
class GTRE(nn.Module):
    """Graph-Temporal Relational Encoder.

    forward(x, pos) returns a dict with:
        ``segments`` : [B, S, d]  -> z_{s,j} for segment-to-token alignment
        ``z_eeg``    : [B, d]     -> montage-agnostic recording embedding (R^256)
    """

    def __init__(self, cfg: GTREConfig | None = None):
        super().__init__()
        self.cfg = cfg or GTREConfig()
        self.temporal = TemporalHierarchicalEncoder(self.cfg)
        self.topo = TopographicalCoordinateEmbedding(self.cfg.d_model,
                                                     self.cfg.spatial_hidden)
        self.layers = nn.ModuleList(
            [GraphTransformerLayer(self.cfg) for _ in range(self.cfg.n_layers)]
        )
        self.norm = nn.LayerNorm(self.cfg.d_model)

    @staticmethod
    def pairwise_distance(pos: torch.Tensor) -> torch.Tensor:
        """Euclidean distance matrix [C, C] from electrode coords [C, 3]."""
        return torch.cdist(pos, pos, p=2)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> dict:
        # x: [B, C, T] raw EEG ; pos: [C, 3] electrode coordinates
        B, C, T = x.shape
        z_macro = self.temporal(x)                 # [B, C, S, d]
        e_pos = self.topo(pos)                      # [C, d]
        z = z_macro + e_pos[None, :, None, :]       # add location embedding

        dist = self.pairwise_distance(pos).to(x.device)  # [C, C]
        S = z.size(2)
        # run the graph transformer over the C electrode tokens, per segment
        z = z.permute(0, 2, 1, 3).reshape(B * S, C, self.cfg.d_model)  # [B*S, C, d]
        for layer in self.layers:
            z = layer(z, dist)
        z = self.norm(z)
        z = z.reshape(B, S, C, self.cfg.d_model)

        segments = z.mean(dim=2)                    # pool channels -> [B, S, d]
        z_eeg = segments.mean(dim=1)                # pool segments -> [B, d]
        return {"segments": segments, "z_eeg": z_eeg, "z_macro": z_macro}
