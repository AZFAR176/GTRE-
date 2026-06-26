r"""Generate rich Masked-DPO preference triplets (t^+, t^-) with the ChatGPT API.

Stage 2 needs, per recording, a *preferred* clinically precise summary ``t^+``
and a *dispreferred* vague summary ``t^-`` that shares the surface semantics but
drops diagnostic precision (paper Sec. "Automated Preference Distillation").

Pipeline per report (faithful to the paper):

  1. **Candidate generation** -- GPT-4 writes several (``--n_candidates``)
     structured, entity-rich summaries of the source report. Each follows the
     clinical reporting structure (background rhythm -> epileptiform findings
     with morphology/frequency/localization -> clinical correlation ->
     impression), ~80-150 words, preserving hedging ("possible", "unclear").
     Optionally (``--mimic_notes_dir``), relevant MIMIC-IV-Note records are
     retrieved and supplied as domain-specific phrasing context -- the paper's
     use of MIMIC narratives to enrich the text resource and improve summary
     quality (style and terminology only; never importing findings, independent
     of EEG).

  2. **Reward-proxy ranking** -- we approximate the paper's lightweight clinical
     reward model by scoring each candidate on *verified clinical-entity
     coverage* (curated vocabulary) + specificity cues (frequencies in Hz,
     laterality, lobe). The top-scoring candidate becomes ``t^+``.

  3. **Controlled perturbation** -- ``t^-`` is derived FROM the chosen ``t^+`` by
     applying ONE of the four **Table 1** strategies, so the pair shares
     semantics but differs only in diagnostic precision:
       1. Entity Underspecification  ("3 Hz spike-and-wave" -> "abnormal activity")
       2. Topographical Deletion     (drop "right temporal" + diagnosis)
       3. Causal De-alignment         (break "following aura / consistent with")
       4. Artifact Disambiguation     (drop cause + entity negation)
     Strategies rotate across the corpus for Table-1-style coverage.

Outputs are paired by base name so the loaders match them to the signals:
    pos_dir/<stem>.txt   (t^+)      neg_dir/<stem>.txt   (t^-)

Setup:
    pip install openai
    export OPENAI_API_KEY=sk-...
Run:
    python -m gtre.make_preferences \
        --reports_dir my_data/reports \
        --pos_dir my_data/t_pos --neg_dir my_data/t_neg \
        --entity_vocab my_data/entity_vocab.txt --model gpt-4o --n_candidates 3 \
        --mimic_notes_dir /path/to/mimic-iv-note/2.2/note   # optional enrichment
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Table-1 perturbation strategies (instruction + worked exemplar from paper)
# --------------------------------------------------------------------------
PERTURBATION_STRATEGIES: Dict[str, Dict[str, str]] = {
    "Entity Underspecification": {
        "instruction":
            "Replace every specific clinical entity (waveform morphology, "
            "frequency in Hz, seizure type) with a vague umbrella term. Keep the "
            "sentence fluent and plausible; do not add new findings.",
        "tpos_example":
            "EEG shows generalized 3 Hz spike-and-wave discharges consistent "
            "with absence seizures.",
        "tneg_example":
            "EEG shows generalized abnormal activity suggestive of possible "
            "seizure events.",
    },
    "Topographical Deletion": {
        "instruction":
            "Delete all localization/topography (e.g. 'right temporal', "
            "'frontal') and the specific diagnosis it supports; end on a "
            "non-committal note such as 'interpretation pending'.",
        "tpos_example":
            "Frequent right temporal sharp waves observed during drowsiness, "
            "suggesting focal epilepsy.",
        "tneg_example":
            "Sharp waves observed during drowsiness; interpretation pending.",
    },
    "Causal De-alignment": {
        "instruction":
            "Break the temporal/causal relationship between events: remove "
            "linking phrases like 'following' or 'consistent with' and state "
            "that the onset is unclear.",
        "tpos_example":
            "Patient exhibited tonic stiffening following aura; consistent with "
            "focal to bilateral tonic-clonic seizure.",
        "tneg_example":
            "Tonic activity observed; unclear if related to focal or "
            "generalized onset.",
    },
    "Artifact Disambiguation": {
        "instruction":
            "Remove the artifact attribution (e.g. medication cause) and any "
            "entity negation (e.g. 'no epileptiform discharges'); replace with a "
            "generic observation such as 'not clearly visualized'.",
        "tpos_example":
            "Burst suppression likely due to Propofol; no epileptiform "
            "discharges noted.",
        "tneg_example":
            "Low amplitude EEG in sedated patient; seizure activity not clearly "
            "visualized.",
    },
}

# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------
TPOS_SYSTEM = (
    "You are a board-certified clinical neurophysiologist writing the formal "
    "interpretation of a routine/long-term EEG. Given a source report, produce "
    "candidate PREFERRED summaries (t_pos) that a downstream model should learn "
    "to favor. Each candidate must:\n"
    "- be a single concise paragraph of ~80-150 words;\n"
    "- follow clinical structure: (a) background rhythm/state, (b) epileptiform "
    "or abnormal findings with EXACT morphology, frequency (Hz), and "
    "localization (lobe + laterality), (c) clinical correlation, (d) a one-line "
    "impression;\n"
    "- preserve EVERY specific entity present or clearly implied in the source, "
    "and NEVER invent findings the source does not support;\n"
    "- preserve hedging words ('possible', 'unclear', 'cannot exclude') exactly "
    "as warranted -- do not become more confident than the source.\n"
    "Candidates should differ in phrasing/completeness, not in invented facts.\n"
    "If REFERENCE CLINICAL PHRASING from related notes is provided, you may adopt "
    "its domain-specific terminology and style, but you must NOT import any "
    "finding from it that the source report does not support.\n"
    "Respond with strict JSON: {\"candidates\": [str, ...]}."
)

TNEG_SYSTEM = (
    "You are degrading a precise EEG summary into a clinically VAGUE variant for "
    "preference training. You will receive a preferred summary (t_pos) and one "
    "perturbation strategy. Produce t_neg by applying ONLY that strategy:\n"
    "- keep t_neg the same length and topic as t_pos and keep it fluent;\n"
    "- share the surface semantics but remove the diagnostic precision the "
    "strategy targets;\n"
    "- do NOT introduce findings absent from t_pos;\n"
    "- preserve hedging words on BOTH sides (never make t_neg more confident).\n"
    "Respond with strict JSON: {\"t_neg\": str}."
)


def _tpos_user(report: str, vocab: Optional[List[str]], n: int,
               mimic_refs: Optional[List[str]] = None) -> str:
    parts = [f"SOURCE EEG REPORT:\n{report.strip()}\n",
             f"Produce {n} candidate t_pos summaries."]
    if vocab:
        parts.append("\nPRIORITY CLINICAL ENTITIES to preserve when supported by "
                     "the source (these drive the downstream pathology mask M):\n"
                     + ", ".join(vocab[:80]))
    if mimic_refs:
        joined = "\n---\n".join(r.strip()[:600] for r in mimic_refs)
        parts.append("\nREFERENCE CLINICAL PHRASING from related MIMIC-IV notes "
                     "(style/terminology only; do NOT import findings):\n" + joined)
    return "\n".join(parts)


def _tneg_user(tpos: str, strategy: str) -> str:
    spec = PERTURBATION_STRATEGIES[strategy]
    return (f"PREFERRED SUMMARY (t_pos):\n{tpos.strip()}\n\n"
            f"PERTURBATION STRATEGY: {strategy}\n{spec['instruction']}\n\n"
            f"WORKED EXAMPLE (for style only, do not copy content):\n"
            f"t_pos: {spec['tpos_example']}\n"
            f"t_neg: {spec['tneg_example']}")


# --------------------------------------------------------------------------
# Reward proxy: rank candidate t_pos by verified clinical specificity
# --------------------------------------------------------------------------
_FREQ_RE = re.compile(r"\b\d+(\.\d+)?\s*hz\b", re.I)
_LATERALITY_RE = re.compile(r"\b(left|right|bilateral|midline|focal|generalized)\b", re.I)
_LOBE_RE = re.compile(r"\b(frontal|temporal|parietal|occipital|central|"
                      r"fronto\w*|centro\w*|temporo\w*)\b", re.I)


def reward_score(text: str, vocab: Optional[List[str]]) -> float:
    """Approximate the clinical reward proxy: reward verified-entity coverage and
    specificity cues (frequencies, laterality, lobe); lightly penalize vague
    filler. Higher = more diagnostically informative."""
    low = text.lower()
    score = 0.0
    if vocab:
        score += 2.0 * sum(1 for e in vocab if e.lower() in low)
    score += 1.5 * len(_FREQ_RE.findall(text))
    score += 1.0 * len(_LATERALITY_RE.findall(text))
    score += 1.0 * len(_LOBE_RE.findall(text))
    for filler in ("abnormal activity", "interpretation pending",
                   "not clearly visualized", "nonspecific", "unremarkable study"):
        if filler in low:
            score -= 1.0
    n_words = max(len(low.split()), 1)
    if n_words < 25:  # too terse to be a rich t_pos
        score -= 1.0
    return score


# --------------------------------------------------------------------------
# OpenAI calls (lazy client) with retry/backoff
# --------------------------------------------------------------------------
def _chat_json(client, model: str, system: str, user: str,
               temperature: float, retries: int = 4) -> dict:
    last = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:  # transient API / JSON parse errors
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"ChatGPT call failed after {retries} retries: {last}")


def generate_tpos(client, model: str, report: str, vocab: Optional[List[str]],
                  n_candidates: int,
                  mimic_refs: Optional[List[str]] = None) -> Tuple[str, List[str]]:
    data = _chat_json(client, model, TPOS_SYSTEM,
                      _tpos_user(report, vocab, n_candidates, mimic_refs),
                      temperature=0.8)
    cands = [c.strip() for c in data.get("candidates", []) if c and c.strip()]
    if not cands:
        raise RuntimeError("no t_pos candidates returned")
    best = max(cands, key=lambda c: reward_score(c, vocab))
    return best, cands


# --------------------------------------------------------------------------
# MIMIC-IV auxiliary text: read the MIMIC-IV-Note CSVs and retrieve domain-
# relevant notes as phrasing context. This is the paper's use of MIMIC
# narratives -- enrich the text resource and improve GPT-summary quality with
# domain-specific linguistic nuance, separate from EEG. It contributes
# style/terminology, never findings.
#
# MIMIC-IV-Note (v2.2): discharge.csv[.gz] / radiology.csv[.gz], columns
#   note_id, subject_id, hadm_id, note_type, note_seq, charttime, storetime, text
# --------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# EEG/neuro-relevant cues used to filter the notes down to the useful subset.
_MIMIC_KEYWORDS = ("eeg", "electroencephalogra", "seizure", "epilep", "ictal",
                   "spike", "sharp wave", "status epilepticus", "convuls",
                   "encephalopath", "nonconvulsive")


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall(text.lower()))


def load_mimic_corpus(notes_dir: Optional[str], max_notes: int = 2000,
                      filter_relevant: bool = True, seed: int = 0) -> List[dict]:
    """Read the MIMIC-IV-Note tables (discharge/radiology) into a lightweight
    in-memory retrieval index. Keeps EEG/neuro-relevant notes by default."""
    if not notes_dir or not os.path.isdir(notes_dir):
        return []
    import pandas as pd  # lazy import
    frames = []
    for name in ("discharge", "radiology"):
        for ext in (".csv", ".csv.gz"):
            p = os.path.join(notes_dir, name + ext)
            if os.path.exists(p):
                frames.append(pd.read_csv(p, usecols=["text"]))
                break
    if not frames:
        return []
    notes = pd.concat(frames, ignore_index=True)
    if filter_relevant:
        pat = "|".join(_MIMIC_KEYWORDS)
        notes = notes[notes["text"].str.contains(pat, case=False, na=False)]
    if max_notes and len(notes) > max_notes:
        notes = notes.sample(max_notes, random_state=seed)
    corpus = []
    for txt in notes["text"].astype(str):
        txt = txt.strip()
        if txt:
            corpus.append({"text": txt, "tok": _tokens(txt)})
    return corpus


def retrieve_mimic(report: str, corpus: List[dict], vocab: Optional[List[str]],
                   k: int = 2) -> List[str]:
    """Top-k MIMIC notes by token overlap with the report (length-normalized),
    boosted by shared curated-entity phrases. Returns note texts."""
    if not corpus:
        return []
    rtok = _tokens(report)
    rlow = report.lower()
    vphrases = [e.lower() for e in (vocab or []) if e.lower() in rlow]
    scored = []
    for note in corpus:
        overlap = len(rtok & note["tok"])
        if overlap == 0:
            continue
        score = overlap / (len(note["tok"]) ** 0.5 + 1.0)
        score += 3.0 * sum(1 for e in vphrases if e in note["text"].lower())
        scored.append((score, note["text"]))
    scored.sort(key=lambda s: s[0], reverse=True)
    return [t for _, t in scored[:k]]


def perturb_to_tneg(client, model: str, tpos: str, strategy: str) -> str:
    data = _chat_json(client, model, TNEG_SYSTEM,
                      _tneg_user(tpos, strategy), temperature=0.7)
    tneg = (data.get("t_neg") or "").strip()
    if not tneg:
        raise RuntimeError("empty t_neg returned")
    return tneg


# --------------------------------------------------------------------------
def load_entity_vocab(path: Optional[str]) -> Optional[List[str]]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser(
        description="Generate rich (t+, t-) preference triplets via the ChatGPT API.")
    ap.add_argument("--reports_dir", required=True,
                    help="Folder of source .txt reports (from gtre/preprocess.py).")
    ap.add_argument("--pos_dir", required=True, help="Output folder for t^+ .txt.")
    ap.add_argument("--neg_dir", required=True, help="Output folder for t^- .txt.")
    ap.add_argument("--entity_vocab", help="Curated clinical-entity vocab .txt.")
    ap.add_argument("--model", default="gpt-4o",
                    help="ChatGPT model (paper uses GPT-4-class).")
    ap.add_argument("--n_candidates", type=int, default=3,
                    help="t^+ candidates per report; best by reward proxy is kept.")
    ap.add_argument("--mimic_notes_dir",
                    help="Optional MIMIC-IV-Note folder (discharge.csv[.gz], "
                         "radiology.csv[.gz]); relevant notes are used as domain-"
                         "specific phrasing context for t^+ (style only).")
    ap.add_argument("--mimic_k", type=int, default=2,
                    help="Number of MIMIC notes to retrieve per report as context.")
    ap.add_argument("--mimic_max", type=int, default=2000,
                    help="Cap MIMIC notes loaded into the retrieval index.")
    ap.add_argument("--mimic_no_filter", action="store_true",
                    help="Index all MIMIC notes (default keeps EEG/neuro-relevant).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N reports (0 = all).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Regenerate even if outputs already exist.")
    ap.add_argument("--save_meta", action="store_true",
                    help="Also write <stem>.json with all candidates + scores.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        from openai import OpenAI  # lazy import
    except ImportError:
        raise SystemExit("pip install openai  (and set OPENAI_API_KEY)")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in your environment.")
    client = OpenAI()

    os.makedirs(args.pos_dir, exist_ok=True)
    os.makedirs(args.neg_dir, exist_ok=True)
    meta_dir = os.path.join(os.path.dirname(args.pos_dir.rstrip("/")) or ".",
                            "preference_meta")
    if args.save_meta:
        os.makedirs(meta_dir, exist_ok=True)

    vocab = load_entity_vocab(args.entity_vocab)
    mimic_corpus = load_mimic_corpus(args.mimic_notes_dir, max_notes=args.mimic_max,
                                     filter_relevant=not args.mimic_no_filter,
                                     seed=args.seed)
    if args.mimic_notes_dir:
        print(f"[mimic] loaded {len(mimic_corpus)} auxiliary notes for phrasing context")
    strategies = list(PERTURBATION_STRATEGIES)

    reports = sorted(glob.glob(os.path.join(args.reports_dir, "**", "*.txt"),
                               recursive=True))
    if args.limit:
        reports = reports[: args.limit]
    if not reports:
        raise SystemExit(f"No .txt reports under {args.reports_dir}")

    done = skipped = failed = 0
    for path in reports:
        stem = os.path.splitext(os.path.basename(path))[0]
        pos_out = os.path.join(args.pos_dir, stem + ".txt")
        neg_out = os.path.join(args.neg_dir, stem + ".txt")
        if not args.overwrite and os.path.exists(pos_out) and os.path.exists(neg_out):
            skipped += 1
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            report = f.read().strip()
        if not report:
            skipped += 1
            continue

        # rotate strategies for Table-1-style coverage across the corpus
        strategy = strategies[done % len(strategies)]
        mimic_refs = retrieve_mimic(report, mimic_corpus, vocab, k=args.mimic_k)
        try:
            tpos, cands = generate_tpos(client, args.model, report, vocab,
                                        args.n_candidates, mimic_refs=mimic_refs)
            tneg = perturb_to_tneg(client, args.model, tpos, strategy)
        except RuntimeError as e:
            print(f"[skip] {stem}: {e}")
            failed += 1
            continue

        _write(pos_out, tpos)
        _write(neg_out, tneg)
        if args.save_meta:
            _write(os.path.join(meta_dir, stem + ".json"), json.dumps({
                "stem": stem, "strategy": strategy, "t_pos": tpos, "t_neg": tneg,
                "candidates": [{"text": c, "reward": reward_score(c, vocab)}
                               for c in cands],
            }, indent=2))
        done += 1
        if done % 25 == 0:
            print(f"  ...generated {done} triplets")

    print(f"Done: {done} generated, {skipped} skipped, {failed} failed.\n"
          f"t^+ -> {args.pos_dir}\nt^- -> {args.neg_dir}"
          + (f"\nmeta -> {meta_dir}" if args.save_meta else ""))


if __name__ == "__main__":
    main()
