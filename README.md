# # GTRE: A Physiological-Semantic Foundation Model for Generalizable EEG Analysis via Preference-Guided Alignment

Reference implementation accompanying **GTRE: A Physiological-Semantic Foundation Model for Generalizable EEG Analysis via Preference-Guided Alignment**.

This repository provides the core model components and training objectives for the Graph-Temporal Relational Encoder (GTRE), including Stage 1 multi-domain EEG-text grounding and Stage 2 Masked-DPO preference refinement.

> **Scope.** This repository does not redistribute TUSZ, TUAB, CHB-MIT, MIMIC-IV, trained checkpoints, expert-validated summaries, or preference pairs. Reproducing the paper results requires access to the original datasets, subject-disjoint splits, and the corresponding text/preference-pair construction protocol.


## What is included

| File | Paper component |
|------|-----------------|
| `gtre/model.py` | GTRE encoder: topographical coordinate embedding, **graph-aware self-attention with the γ geometric bias (Eq. 1)**, temporal hierarchy (`Z_micro` 200 ms → `Z_macro` 5 s ∈ ℝ²⁵⁶), L=8/H=8 transformer producing `z_eeg`. |
| `gtre/text_encoder.py` | Frozen BioClinicalBERT + projection head; per-entity embeddings for segment-to-token alignment. |
| `gtre/data.py` | Directory-based datasets (`EEGTextPairDataset`, `PreferenceDataset`) that pair files by name, plus signal/text/montage loaders and collate fns. |
| `gtre/preprocess.py` | EEG prep: EDF (TUSZ/TUAB) band-pass + resample + z-score → `[C,T]` `.npy`, plus the 10-20 montage and starter entity vocab. |
| `gtre/make_preferences.py` | GPT-4 (ChatGPT API) generation of Masked-DPO triplets: ranked structured `t⁺` (reward-proxy over entity coverage) + a paired vague `t⁻` via the four Table-1 perturbation strategies. |
| `gtre/losses.py` | **Segmental CLIP loss (Eq. 2)**, domain-invariant prototype KL (`L_dom`), **Masked-DPO loss (Eq. 3)**, consistency prototype loss (`L_proto`). |
| `gtre/train_stage1.py` | Stage 1: multi-domain grounding pipeline. |
| `gtre/train_stage2.py` | Stage 2: Masked-DPO refinement (frozen reference `π_ref`, reward-proxy stub). |
| `gtre/evaluate.py` | Results protocols: single-layer MLP head, few-shot (5%), full fine-tuning (SOTA), zero-shot (cosine to entity embeddings). |
| `configs/default.yaml` | Hyperparameters from the Implementation Details. |

## Usage

### 1. Install

```bash
pip install -r requirements.txt   # torch, numpy, transformers, pyyaml
```

### 2. Prepare your own data into folders

```
my_data/
├── signals/                # EEG arrays, shape [C, T]  (.npy/.npz/.pt/.csv/.edf)
│   ├── TUSZ/rec_0001.npy    # optional subfolder = domain id for L_dom
│   └── TUAB/rec_0002.npy
├── reports/                # Stage-1 text: rec_0001.txt, rec_0002.txt ...
├── t_pos/                  # Stage-2 preferred summaries t⁺ : rec_0001.txt ...
├── t_neg/                  # Stage-2 dispreferred summaries t⁻: rec_0001.txt ...
├── montage.csv             # name,x,y,z  (electrode 3D coords)
└── entity_vocab.txt        # one curated clinical entity per line
```

Pairing rule: `signals/.../rec_0001.npy` ↔ `reports/rec_0001.txt` (and
`t_pos/rec_0001.txt`, `t_neg/rec_0001.txt`).

`gtre/preprocess.py` builds the
`signals/`, `reports/`, `montage.csv`, and `entity_vocab.txt` from raw EDFs.
Edit the placeholder `RAW_*` paths at the top of that file (`RAW_TUSZ_DIR`,
`RAW_TUAB_DIR`) to point at your local copies, then:

```bash
python -m gtre.preprocess --out_dir my_data
```

It picks the standard 10-20 channels (real coordinates from MNE, falling back to
a built-in table), band-passes (0.5-45 Hz), resamples to 250 Hz, z-scores per
channel, and writes one `.npy` + `.txt` per recording. Pass `--entity_vocab
my_vocab.txt` to use your own clinical-entity vocabulary instead of the starter
list.

**Stage-2 preference pairs.** `gtre/make_preferences.py` generates the `t_pos/`
and `t_neg/` triplets from `reports/` using the ChatGPT API (GPT-4-class), in
three steps per report: (1) generate several structured, entity-rich candidate
summaries (background → epileptiform findings with morphology/frequency/
localization → correlation → impression); (2) rank them with a lightweight
**reward proxy** that scores verified clinical-entity coverage + specificity
cues (Hz, laterality, lobe) and keeps the best as `t⁺`; (3) derive a paired
vague `t⁻` by applying one of the four Table-1 perturbation strategies (entity
underspecification, topographical deletion, causal de-alignment, artifact
disambiguation), rotated for coverage. Hedging terms ("possible", "unclear")
are preserved on both sides.

```bash
pip install openai
export OPENAI_API_KEY=sk-...
python -m gtre.make_preferences \
    --reports_dir my_data/reports \
    --pos_dir my_data/t_pos --neg_dir my_data/t_neg \
    --entity_vocab my_data/entity_vocab.txt \
    --model gpt-4o --n_candidates 3 --save_meta \
    --mimic_notes_dir /path/to/mimic-iv-note/2.2/note   # optional enrichment
```

Pass `--mimic_notes_dir` (the MIMIC-IV-Note folder with `discharge.csv[.gz]` /
`radiology.csv[.gz]`) to use the auxiliary notes: relevant, EEG/neuro-filtered
records are retrieved per report (token overlap + curated entities) and given to
GPT-4 as domain-specific **phrasing/terminology context** (style only, no
imported findings). This is the paper's use of MIMIC narratives to enrich the
clinical-text resource and improve summary quality, independent of EEG.

(`--save_meta` also dumps every candidate + reward score to `preference_meta/`
for auditing; `--limit N` does a small trial run; reruns skip existing pairs
unless `--overwrite`.)

### 3. Stage 1 — multi-domain grounding

```bash
python -m gtre.train_stage1 \
    --signals_dir my_data/signals --texts_dir my_data/reports \
    --montage my_data/montage.csv --entity_vocab my_data/entity_vocab.txt \
    --epochs 20 --batch_size 1024 --device cuda
```

Builds the GTRE encoder + frozen BioClinicalBERT text encoder and optimizes the
segmental-CLIP loss + domain-invariant prototype KL (`L_stage1`). The trained
encoder is saved to `stage1_encoder.pt` (override with `--out_ckpt`).

### 4. Stage 2 — Masked-DPO refinement

```bash
python -m gtre.train_stage2 \
    --signals_dir my_data/signals --pos_dir my_data/t_pos \
    --neg_dir my_data/t_neg --montage my_data/montage.csv \
    --entity_vocab my_data/entity_vocab.txt \
    --stage1_ckpt stage1_encoder.pt --epochs 10 --device cuda
```

Initializes `π_θ` from the Stage-1 checkpoint, keeps a frozen `π_ref`, and
optimizes the Masked-DPO preference loss over similarity margins.

### 5. Evaluate

`gtre/evaluate.py` provides the four protocols from the paper — single-layer MLP
linear probe, few-shot fine-tuning with 5% labels, full fine-tuning (SOTA), and
zero-shot (cosine to entity embeddings) — which you call on a downstream
labeled split.

## Data access

TUSZ, TUAB, and MIMIC-IV are **access-controlled** and cannot be redistributed
here. Obtain them from their official sources under the respective data-use
agreements; this repo provides the model, objectives, and (to follow) the
de-identified pairing/preprocessing protocols.

## Authors

- **Mohd Azfar** — Indian Institute of Science, Bengaluru, India · azfar45@gmail.com
- **Izhar Dad Khan** — Basque Center on Cognition, Brain and Language, San Sebastián, Spain · i.khan@bcbl.eu

Both authors contributed equally to this work.

## License

Released under the [MIT License](LICENSE).

