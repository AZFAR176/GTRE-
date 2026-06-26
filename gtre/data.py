"""Data loading for GTRE (directory-based).

You point the datasets at **folders** and they pair files by base name (stem):

Stage 1 (EEG-text pairs) -- ``EEGTextPairDataset``
    signals_dir/  ...  EEG files            ([C, T] : .npy/.npz/.pt/.csv/.edf)
    texts_dir/    ...  matching .txt reports (same stem as the signal)
    Optional: organise signals into per-dataset subfolders
              signals_dir/TUSZ/*.npy, signals_dir/TUAB/*.npy
              -> the subfolder name becomes the domain id (for L_dom).

Stage 2 (preference triplets) -- ``PreferenceDataset``
    signals_dir/  ...  EEG files
    pos_dir/      ...  preferred summaries  t^+  (.txt, same stem)
    neg_dir/      ...  dispreferred summaries t^- (.txt, same stem)
    The pathology mask M is derived from the curated entity vocabulary.

A signal ``rec_0001.npy`` is paired with ``rec_0001.txt`` in each text folder.
"""

from __future__ import annotations

import csv
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

SIGNAL_EXTS = (".npy", ".npz", ".pt", ".csv", ".txt", ".edf")
TEXT_EXTS = (".txt",)


# ---------------------------------------------------------------------------
# Low-level loaders
# ---------------------------------------------------------------------------
def load_signal(path: str, n_time: Optional[int] = None,
                n_channels: Optional[int] = None) -> torch.Tensor:
    """Load an EEG recording as a float tensor of shape [C, T]."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".npz":
        npz = np.load(path)
        arr = npz[npz.files[0]]
    elif ext == ".pt":
        arr = torch.load(path, map_location="cpu")
        arr = arr.numpy() if isinstance(arr, torch.Tensor) else np.asarray(arr)
    elif ext in (".csv", ".txt"):
        arr = np.loadtxt(path, delimiter=",")
    elif ext == ".edf":
        import mne  # optional dependency
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        arr = raw.get_data()
    else:
        raise ValueError(f"Unsupported signal format: {ext}")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if n_channels is not None and arr.shape[0] != n_channels and arr.shape[1] == n_channels:
        arr = arr.T
    x = torch.from_numpy(np.ascontiguousarray(arr))
    if n_time is not None:
        x = _fix_length(x, n_time)
    return x


def _fix_length(x: torch.Tensor, n_time: int) -> torch.Tensor:
    """Crop or zero-pad the time axis (last dim) to ``n_time``."""
    T = x.size(-1)
    if T == n_time:
        return x
    if T > n_time:
        return x[..., :n_time]
    pad = x.new_zeros(*x.shape[:-1], n_time - T)
    return torch.cat([x, pad], dim=-1)


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def load_entity_vocab(path: Optional[str]) -> Optional[List[str]]:
    """Curated clinical-entity vocabulary (one entity per line). Drives mask M
    and entity extraction; distinct from the learnable prototypes P."""
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


_DELIMS = ("|", ";", "\n")


def to_entities(text: str, vocab: Optional[Sequence[str]] = None,
                max_entities: int = 16) -> List[str]:
    """Turn a report into a list of clinical entities. If a curated ``vocab`` is
    given, keep vocabulary terms present in the text; else split on delimiters."""
    text = text or ""
    if vocab:
        hits = [e for e in vocab if e.lower() in text.lower()]
        if hits:
            return hits[:max_entities]
    parts = [text]
    for d in _DELIMS:
        parts = sum((p.split(d) for p in parts), [])
    ents = [p.strip() for p in parts if p.strip()]
    return ents[:max_entities] if ents else [text.strip() or "n/a"]


def derive_mask(entities: Sequence[str],
                vocab: Optional[Sequence[str]]) -> List[int]:
    """Pathology-aware mask M in {0,1}: 1 for curated diagnostic entities."""
    if not vocab:
        return [1] * len(entities)
    low = {v.lower() for v in vocab}
    return [1 if e.lower() in low else 0 for e in entities]


def load_montage(path: str) -> Tuple[torch.Tensor, List[str]]:
    """Load electrode coordinates from a CSV with columns name,x,y,z.
    Returns (pos[C,3] float tensor, channel_names)."""
    names, coords = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            names.append(row.get("name", str(len(names))))
            coords.append([float(row["x"]), float(row["y"]), float(row["z"])])
    return torch.tensor(coords, dtype=torch.float32), names


# ---------------------------------------------------------------------------
# Directory indexing / pairing by base name (stem)
# ---------------------------------------------------------------------------
def _list_dir(root: str, exts) -> List[Tuple[str, str, str]]:
    """Recursively list files under ``root``.
    Returns sorted tuples (domain, stem, fullpath); domain = first subfolder."""
    if not root or not os.path.isdir(root):
        raise FileNotFoundError(f"Directory not found: {root}")
    out = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in exts:
                rel = os.path.relpath(dirpath, root)
                domain = "0" if rel == "." else rel.split(os.sep)[0]
                out.append((domain, os.path.splitext(fn)[0],
                            os.path.join(dirpath, fn)))
    if not out:
        raise FileNotFoundError(f"No files with {exts} under {root}")
    return sorted(out, key=lambda t: (t[0], t[1]))


def _text_lookup(text_dir: str) -> dict:
    """Map stem -> text file path for a (possibly nested) text directory."""
    return {stem: path for _, stem, path in _list_dir(text_dir, TEXT_EXTS)}


# ---------------------------------------------------------------------------
# Stage 1 dataset: EEG-text pairs
# ---------------------------------------------------------------------------
class EEGTextPairDataset(Dataset):
    def __init__(self, signals_dir: str, texts_dir: str, n_time: int = 1250,
                 n_channels: int = 20, entity_vocab: Optional[Sequence[str]] = None):
        self.signals = _list_dir(signals_dir, SIGNAL_EXTS)   # [(domain, stem, path)]
        self.texts = _text_lookup(texts_dir)
        self.n_time, self.n_channels = n_time, n_channels
        self.vocab = entity_vocab
        missing = [s for _, s, _ in self.signals if s not in self.texts]
        if missing:
            raise ValueError(f"{len(missing)} signals have no matching .txt "
                             f"(e.g. '{missing[0]}') in {texts_dir}")
        domains = sorted({d for d, _, _ in self.signals})
        self.domain_to_id = {d: i for i, d in enumerate(domains)}

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, i):
        domain, stem, sig_path = self.signals[i]
        eeg = load_signal(sig_path, self.n_time, self.n_channels)
        entities = to_entities(load_text(self.texts[stem]), self.vocab)
        return eeg, entities, self.domain_to_id[domain]


def collate_pairs(batch):
    eeg = torch.stack([b[0] for b in batch])
    entities = [b[1] for b in batch]
    domain = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return eeg, entities, domain


# ---------------------------------------------------------------------------
# Stage 2 dataset: preference triplets (signal, t^+, t^-)
# ---------------------------------------------------------------------------
class PreferenceDataset(Dataset):
    def __init__(self, signals_dir: str, pos_dir: str, neg_dir: str,
                 n_time: int = 1250, n_channels: int = 20,
                 entity_vocab: Optional[Sequence[str]] = None):
        self.signals = _list_dir(signals_dir, SIGNAL_EXTS)
        self.pos = _text_lookup(pos_dir)
        self.neg = _text_lookup(neg_dir)
        self.n_time, self.n_channels = n_time, n_channels
        self.vocab = entity_vocab
        for name, look in (("t+", self.pos), ("t-", self.neg)):
            miss = [s for _, s, _ in self.signals if s not in look]
            if miss:
                raise ValueError(f"{len(miss)} signals have no matching {name} "
                                 f".txt (e.g. '{miss[0]}').")

    def __len__(self):
        return len(self.signals)

    def _ent_mask(self, path):
        ents = to_entities(load_text(path), self.vocab)
        return ents, derive_mask(ents, self.vocab)

    def __getitem__(self, i):
        _, stem, sig_path = self.signals[i]
        eeg = load_signal(sig_path, self.n_time, self.n_channels)
        ents_pos, mask_pos = self._ent_mask(self.pos[stem])
        ents_neg, mask_neg = self._ent_mask(self.neg[stem])
        return eeg, ents_pos, ents_neg, mask_pos, mask_neg


def collate_preferences(batch):
    eeg = torch.stack([b[0] for b in batch])
    return (eeg, [b[1] for b in batch], [b[2] for b in batch],
            [b[3] for b in batch], [b[4] for b in batch])


def pad_mask(mask_lists: Sequence[Sequence[int]], width: int,
             device="cpu") -> torch.Tensor:
    """Pad a list of variable-length 0/1 masks to [B, width]."""
    out = torch.zeros(len(mask_lists), width, device=device)
    for i, m in enumerate(mask_lists):
        m = list(m)[:width]
        out[i, : len(m)] = torch.tensor(m, dtype=torch.float, device=device)
    return out
