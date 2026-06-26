r"""Raw -> training-ready data preparation for GTRE.

This turns the *raw* corpora into the on-disk layout that ``gtre/data.py``
expects (see README "Prepare your own data into folders"):

    OUT_DIR/
    ├── signals/<DOMAIN>/<stem>.npy   # [C, T] float32, preprocessed EEG (TUSZ/TUAB)
    ├── reports/<stem>.txt            # Stage-1 grounding text per EEG recording
    ├── montage.csv                   # name,x,y,z electrode coordinates
    └── entity_vocab.txt              # curated clinical entities (edit to taste)

EEG preprocessing (per recording):
    EDF -> pick standard 10-20 channels -> band-pass [L_FREQ, H_FREQ] Hz
        -> resample to TARGET_FS -> z-score per channel -> [C, T] .npy

TUSZ and TUAB provide the EEG (their subfolder name -> domain id for L_dom).
The auxiliary MIMIC-IV-Note text is handled separately, where it is consumed:
see ``gtre/make_preferences.py --mimic_notes_dir`` (summary-phrasing enrichment).

NOTE: the corpora are access-controlled, so the RAW_* paths below are
**placeholders** — set them to your local copies before running. Heavy deps
(mne/scipy) are imported lazily so the module still imports without them.

Run:
    python -m gtre.preprocess --out_dir my_data
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# CONFIG -- EDIT THESE PLACEHOLDER PATHS to point at your local raw corpora
# ---------------------------------------------------------------------------
# TUH EEG Seizure Corpus (TUSZ) and TUH Abnormal Corpus (TUAB): trees of .edf
# recordings, each typically with an adjacent annotation file.
RAW_TUSZ_DIR = "/path/to/tuh_eeg_seizure/edf"      # *.edf + *.csv_bi / *.tse_bi
RAW_TUAB_DIR = "/path/to/tuh_eeg_abnormal/edf"      # *.edf + normal/abnormal label

# Preprocessing hyperparameters (match Implementation Details / your setup).
TARGET_FS = 250          # Hz after resampling
L_FREQ, H_FREQ = 0.5, 45.0   # band-pass cutoffs (Hz)
SEGMENT_SECONDS = 60         # crop/keep this many seconds per recording (None = full)

# Channels we keep (and their order). Coordinates below are an OFFLINE FALLBACK
# (approximate unit-sphere); when mne is installed, write_montage() pulls the
# real 'standard_1020' positions instead. Both the channel set and the montage
# feed the model's topographical coordinate embedding, so this is required.
# Axis convention (MNE head frame): +x = right, +y = anterior (nasion),
# +z = superior; left hemisphere x<0, posterior y<0, vertex Cz = (0,0,1).
STANDARD_1020: Dict[str, Tuple[float, float, float]] = {
    "Fp1": (-0.31, 0.95, -0.03), "Fp2": (0.31, 0.95, -0.03),
    "F7": (-0.81, 0.59, -0.03), "F3": (-0.55, 0.67, 0.50),
    "Fz": (0.00, 0.72, 0.69), "F4": (0.55, 0.67, 0.50),
    "F8": (0.81, 0.59, -0.03), "T3": (-0.99, 0.00, -0.03),
    "C3": (-0.71, 0.00, 0.71), "Cz": (0.00, 0.00, 1.00),
    "C4": (0.71, 0.00, 0.71), "T4": (0.99, 0.00, -0.03),
    "T5": (-0.81, -0.59, -0.03), "P3": (-0.55, -0.67, 0.50),
    "Pz": (0.00, -0.72, 0.69), "P4": (0.55, -0.67, 0.50),
    "T6": (0.81, -0.59, -0.03), "O1": (-0.31, -0.95, -0.03),
    "O2": (0.31, -0.95, -0.03),
}
CHANNELS = list(STANDARD_1020.keys())


# ---------------------------------------------------------------------------
# Montage + vocab writers
# ---------------------------------------------------------------------------
def _standard_1020_coords() -> Dict[str, Tuple[float, float, float]]:
    """Real 10-20 coordinates from MNE if available, else the offline fallback.
    MNE uses modern names (T7/T8/P7/P8) for our T3/T4/T5/T6."""
    try:
        import mne
        pos = mne.channels.make_standard_montage(
            "standard_1020").get_positions()["ch_pos"]
        alias = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
        out = {ch: tuple(pos[alias.get(ch, ch)]) for ch in CHANNELS
               if alias.get(ch, ch) in pos}
        if len(out) == len(CHANNELS):
            return out
    except Exception:
        pass
    return STANDARD_1020


def write_montage(out_dir: str) -> str:
    path = os.path.join(out_dir, "montage.csv")
    coords = _standard_1020_coords()
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "x", "y", "z"])
        w.writeheader()
        for name in CHANNELS:
            x, y, z = coords[name]
            n = (x * x + y * y + z * z) ** 0.5 or 1.0   # unit-normalize geometry
            w.writerow({"name": name, "x": x / n, "y": y / n, "z": z / n})
    return path


# Starter clinical-entity vocabulary written to entity_vocab.txt; it drives the
# pathology mask M, entity alignment, and the make_preferences reward proxy.
# Edit it for your label space, or pass --entity_vocab to supply your own file.
DEFAULT_VOCAB = [
    "seizure", "focal seizure", "generalized seizure", "absence seizures",
    "3 Hz spike-and-wave", "spike-and-wave", "sharp waves", "spikes",
    "right temporal sharp waves", "left temporal sharp waves",
    "focal to bilateral tonic-clonic", "tonic-clonic", "epileptiform discharges",
    "focal epilepsy", "status epilepticus", "burst suppression", "aura",
    "abnormal", "normal", "slowing", "polymorphic delta", "PLEDs", "GPEDs",
]


def write_entity_vocab(out_dir: str, src: Optional[str] = None) -> str:
    """Write entity_vocab.txt. If ``src`` is given, copy that file's entities;
    otherwise emit the editable DEFAULT_VOCAB starter list."""
    path = os.path.join(out_dir, "entity_vocab.txt")
    if src and os.path.exists(src):
        with open(src, "r", encoding="utf-8") as f:
            terms = [ln.strip() for ln in f if ln.strip()]
    else:
        terms = DEFAULT_VOCAB
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(terms) + "\n")
    return path


# ---------------------------------------------------------------------------
# EEG preprocessing
# ---------------------------------------------------------------------------
def preprocess_eeg(edf_path: str) -> Optional[np.ndarray]:
    """EDF -> [C, T] float32 (picked 10-20 channels, filtered, resampled, z-scored).
    Returns None if the recording lacks the required channels."""
    import mne  # lazy import

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    raw.rename_channels({c: _clean_ch(c) for c in raw.ch_names})
    present = [c for c in CHANNELS if c in raw.ch_names]
    if len(present) < len(CHANNELS) // 2:
        return None  # too few standard channels -- skip
    raw.pick_channels(present, ordered=True)
    raw.filter(L_FREQ, H_FREQ, verbose="ERROR")
    if raw.info["sfreq"] != TARGET_FS:
        raw.resample(TARGET_FS, verbose="ERROR")

    x = raw.get_data().astype(np.float32)              # [C, T]
    if SEGMENT_SECONDS:
        x = x[:, : int(SEGMENT_SECONDS * TARGET_FS)]
    # re-insert any missing channels as zeros so [C] stays consistent
    if present != CHANNELS:
        full = np.zeros((len(CHANNELS), x.shape[1]), dtype=np.float32)
        idx = {c: i for i, c in enumerate(present)}
        for i, c in enumerate(CHANNELS):
            if c in idx:
                full[i] = x[idx[c]]
        x = full
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-6
    return (x - mu) / sd


def _clean_ch(name: str) -> str:
    """Normalize TUH channel names like 'EEG FP1-REF' -> 'Fp1'."""
    n = name.upper().replace("EEG", "").replace("-REF", "").replace("-LE", "").strip()
    alias = {"FP1": "Fp1", "FP2": "Fp2", "T7": "T3", "T8": "T4",
             "P7": "T5", "P8": "T6"}
    return alias.get(n, n.capitalize() if len(n) > 1 else n)


# ---------------------------------------------------------------------------
# Per-domain builders (TUSZ / TUAB)
# ---------------------------------------------------------------------------
# TUSZ term codes -> readable clinical entities (NEDC annotation guidelines).
TUSZ_LABELS = {
    "bckg": "background", "seiz": "seizure",
    "fnsz": "focal nonspecific seizure", "gnsz": "generalized nonspecific seizure",
    "spsz": "simple partial seizure", "cpsz": "complex partial seizure",
    "absz": "absence seizures", "tnsz": "tonic seizure", "cnsz": "clonic seizure",
    "tcsz": "focal to bilateral tonic-clonic", "atsz": "atonic seizure",
    "mysz": "myoclonic seizure", "nesz": "nonepileptic seizure",
}


def _parse_tusz_labels(path: str) -> List[str]:
    """Read a TUSZ annotation file and return the distinct seizure-type labels.

    Handles both layouts robustly: ``.tse/.tse_bi`` are space-delimited
    ``start stop label prob``; ``.csv/.csv_bi`` (v2.0.0+) are comma-delimited
    ``channel,start_time,stop_time,label,confidence``. In BOTH the label is the
    second-to-last field, so we key off that and ignore leading columns/headers.
    """
    labels = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#") or ln.lower().startswith(("version", "channel")):
                continue
            toks = ln.split(",") if "," in ln else ln.split()
            if len(toks) < 4:
                continue
            labels.append(toks[-2].strip().lower())   # label = second-to-last
    seen = [l for l in dict.fromkeys(labels) if l and l != "bckg"]
    return seen


def _report_from_annotation(edf_path: str, domain: str) -> str:
    """Build the grounding report text for a recording from its sidecar labels.

    TUSZ: read the adjacent .csv / .csv_bi / .tse / .tse_bi and list the
    seizure types present (mapped to readable entities). TUAB: the label is
    encoded in the path (.../abnormal/... or .../normal/...).
    """
    base = edf_path[:-4] if edf_path.lower().endswith(".edf") else edf_path
    for ext in (".csv", ".csv_bi", ".tse", ".tse_bi"):
        if os.path.exists(base + ext):
            try:
                codes = _parse_tusz_labels(base + ext)
            except Exception:
                codes = []
            if codes:
                terms = sorted({TUSZ_LABELS.get(c, c) for c in codes})
                return " | ".join(terms)
            if domain == "TUSZ":
                return "no electrographic seizures | background activity"
            break
    # TUAB (and TUSZ without a sidecar): fall back to the path-encoded label.
    return "abnormal" if "abnormal" in edf_path.lower() else "normal"


def build_domain(raw_dir: str, domain: str, out_dir: str) -> int:
    """Preprocess all EDFs under ``raw_dir`` into signals/<domain>/ + reports/."""
    sig_dir = os.path.join(out_dir, "signals", domain)
    rep_dir = os.path.join(out_dir, "reports")
    os.makedirs(sig_dir, exist_ok=True)
    os.makedirs(rep_dir, exist_ok=True)

    edfs = sorted(glob.glob(os.path.join(raw_dir, "**", "*.edf"), recursive=True))
    if not edfs:
        print(f"[{domain}] no .edf under {raw_dir} (set RAW_{domain}_DIR) -- skipping")
        return 0

    n = 0
    for i, edf in enumerate(edfs):
        x = preprocess_eeg(edf)
        if x is None:
            continue
        stem = f"{domain.lower()}_{i:05d}"
        np.save(os.path.join(sig_dir, stem + ".npy"), x)
        _write_text(os.path.join(rep_dir, stem + ".txt"),
                    _report_from_annotation(edf, domain))
        n += 1
    print(f"[{domain}] wrote {n}/{len(edfs)} recordings")
    return n


# ---------------------------------------------------------------------------
def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text or "n/a")


def main():
    ap = argparse.ArgumentParser(description="Prepare GTRE EEG data from raw corpora.")
    ap.add_argument("--out_dir", required=True, help="Output dataset root.")
    ap.add_argument("--entity_vocab", help="Use your own entity vocab .txt "
                                           "instead of the built-in starter list.")
    ap.add_argument("--skip_tusz", action="store_true")
    ap.add_argument("--skip_tuab", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    write_montage(args.out_dir)
    write_entity_vocab(args.out_dir, src=args.entity_vocab)

    if not args.skip_tusz:
        build_domain(RAW_TUSZ_DIR, "TUSZ", args.out_dir)
    if not args.skip_tuab:
        build_domain(RAW_TUAB_DIR, "TUAB", args.out_dir)

    print(f"\nDone. Train with:\n"
          f"  python -m gtre.train_stage1 --signals_dir {args.out_dir}/signals "
          f"--texts_dir {args.out_dir}/reports "
          f"--montage {args.out_dir}/montage.csv "
          f"--entity_vocab {args.out_dir}/entity_vocab.txt")


if __name__ == "__main__":
    main()
