from __future__ import annotations

import torch
import torch.nn.functional as F


# ===========================================================================
# Stage 1
# ===========================================================================
def segmental_clip_loss(z_seg: torch.Tensor, z_ent: torch.Tensor,
                        tau: float = 0.07) -> torch.Tensor:
    """Fine-grained Segment-to-Token CLIP loss (Eq. 2).

    For a pair (EEG_n, Text_{n'}) the alignment score is the *max* cosine
    similarity over all (segment j, entity k) pairs:
        A_{n,n'} = max_{j,k} cos(z_{s,j}^{(n)}, z_{e,k}^{(n')}).
    A symmetric InfoNCE over the batch then pulls each EEG toward its own report.

    Args:
        z_seg: [B, S, d]  EEG segment embeddings z_{s,j}
        z_ent: [B, V, d]  clinical entity embeddings z_{e,k}
    """
    z_seg = F.normalize(z_seg, dim=-1)
    z_ent = F.normalize(z_ent, dim=-1)
    # pairwise segment-entity cosine sims: [B(eeg), B(text), S, V]
    sims = torch.einsum("isd,jvd->ijsv", z_seg, z_ent)
    A = sims.flatten(start_dim=2).max(dim=-1).values        # max_{j,k} -> [B, B]
    logits = A / tau
    targets = torch.arange(A.size(0), device=A.device)
    # symmetric (EEG->text and text->EEG), as in CLIP
    loss = 0.5 * (F.cross_entropy(logits, targets)
                  + F.cross_entropy(logits.t(), targets))
    return loss


def _soft_assign(z: torch.Tensor, prototypes: torch.Tensor,
                 tau: float) -> torch.Tensor:
    """q(z) = softmax(z P^T / tau).  z: [N, d], P: [K, d] -> [N, K]."""
    return torch.softmax(z @ prototypes.t() / tau, dim=-1)


def prototype_kl_regularizer(z: torch.Tensor, domain_ids: torch.Tensor,
                             prototypes: torch.Tensor, tau: float = 0.1,
                             eps: float = 1e-8) -> torch.Tensor:
    """Domain-invariant prototype regularization:
        L_dom = sum_m KL( qbar^m || qbar ),
    where qbar^m is the mean soft-assignment within domain m and qbar the global
    mean. ``z`` are segment embeddings (flattened over B and S), ``domain_ids``
    the per-row domain index m.
    """
    q = _soft_assign(z, prototypes, tau)                    # [N, K]
    qbar = q.mean(dim=0, keepdim=True)                      # [1, K] global mean
    loss = z.new_zeros(())
    for m in torch.unique(domain_ids):
        qm = q[domain_ids == m].mean(dim=0, keepdim=True)   # [1, K]
        loss = loss + (qm * (torch.log(qm + eps) - torch.log(qbar + eps))).sum()
    return loss


def stage1_loss(z_seg: torch.Tensor, z_ent: torch.Tensor,
                domain_ids: torch.Tensor, prototypes: torch.Tensor,
                lambda_dom: float = 0.1, tau_clip: float = 0.07,
                tau_proto: float = 0.1) -> dict:
    """Complete Stage 1 objective:
        L_stage1 = mean_m L_seg-clip^m + lambda_dom * L_dom.
    """
    clip = segmental_clip_loss(z_seg, z_ent, tau=tau_clip)
    z_flat = z_seg.reshape(-1, z_seg.size(-1))
    dom_flat = domain_ids.repeat_interleave(z_seg.size(1))
    dom = prototype_kl_regularizer(z_flat, dom_flat, prototypes, tau=tau_proto)
    total = clip + lambda_dom * dom
    return {"loss": total, "clip": clip.detach(), "dom": dom.detach()}


# ===========================================================================
# Stage 2 : Masked-DPO
# ===========================================================================
def masked_similarity(z_eeg: torch.Tensor, z_text_entities: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
    """Normalized dot-product similarity s = cos(z_e, M (x) z_t).

    The pathology mask M in {0,1}^V keeps only diagnostic entity tokens; the
    masked text embedding is their (mask-weighted) mean.
    Args:
        z_eeg:           [B, d]     EEG embedding from GTRE
        z_text_entities: [B, V, d]  per-entity text embeddings z_t
        mask:            [B, V]     pathology-aware mask M
    Returns:
        s: [B] cosine similarities.
    """
    m = mask.unsqueeze(-1).float()                          # [B, V, 1]
    masked_text = (z_text_entities * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-8)
    return F.cosine_similarity(z_eeg, masked_text, dim=-1)  # [B]


def masked_dpo_loss(s_theta_pos: torch.Tensor, s_theta_neg: torch.Tensor,
                    s_ref_pos: torch.Tensor, s_ref_neg: torch.Tensor,
                    delta: torch.Tensor, beta: float = 0.1,
                    lam: float = 0.3) -> torch.Tensor:
    """Similarity-scaled Masked-DPO objective (Eq. 3):

        L = - log sigma( beta [ (s_theta^+ - s_ref^+) - (s_theta^- - s_ref^-) ]
                         - lambda (1 - delta) ).

    The adaptive term lambda(1 - delta) softens supervision when t^+ ~= t^-
    (delta = cos(z_{t^+}, z_{t^-}) -> 1).
    """
    margin = (s_theta_pos - s_ref_pos) - (s_theta_neg - s_ref_neg)
    logits = beta * margin - lam * (1.0 - delta)
    return -F.logsigmoid(logits).mean()


def consistency_proto_loss(z_old: torch.Tensor, z_new: torch.Tensor,
                           prototypes: torch.Tensor, tau: float = 0.1,
                           eps: float = 1e-8) -> torch.Tensor:
    """L_proto = - sum_k p_old^(k) log p_new^(k)  (cross-entropy of soft
    prototype assignments), preserving the Stage-1 domain-invariant structure.
    ``z_old`` is from the frozen reference, ``z_new`` from the trained encoder.
    """
    p_old = _soft_assign(z_old.detach(), prototypes, tau)
    p_new = _soft_assign(z_new, prototypes, tau)
    return -(p_old * torch.log(p_new + eps)).sum(dim=-1).mean()
