"""GTRE: a physiological-semantic EEG foundation model (reference implementation)."""

from .model import (GTRE, GTREConfig, TopographicalCoordinateEmbedding,
                    TemporalHierarchicalEncoder, GraphAwareAttention,
                    GraphTransformerLayer)
from .text_encoder import ClinicalTextEncoder
from .losses import (segmental_clip_loss, prototype_kl_regularizer, stage1_loss,
                     masked_similarity, masked_dpo_loss, consistency_proto_loss)
from .data import (load_signal, load_text, load_entity_vocab, to_entities,
                   derive_mask, load_montage, EEGTextPairDataset, collate_pairs,
                   PreferenceDataset, collate_preferences, pad_mask)
from .evaluate import (MLPHead, linear_probe, few_shot, full_finetune, zero_shot)
from .train_stage1 import train_stage1
from .train_stage2 import train_stage2, RewardProxy

__all__ = [
    # model
    "GTRE", "GTREConfig", "TopographicalCoordinateEmbedding",
    "TemporalHierarchicalEncoder", "GraphAwareAttention", "GraphTransformerLayer",
    # text encoder
    "ClinicalTextEncoder",
    # losses
    "segmental_clip_loss", "prototype_kl_regularizer", "stage1_loss",
    "masked_similarity", "masked_dpo_loss", "consistency_proto_loss",
    # data
    "load_signal", "load_text", "load_entity_vocab", "to_entities", "derive_mask",
    "load_montage", "EEGTextPairDataset", "collate_pairs", "PreferenceDataset",
    "collate_preferences", "pad_mask",
    # evaluation
    "MLPHead", "linear_probe", "few_shot", "full_finetune", "zero_shot",
    # training
    "train_stage1", "train_stage2", "RewardProxy",
]
