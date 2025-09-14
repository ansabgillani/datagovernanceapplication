#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute WER per anonymization method by comparing:
  - REF (original transcripts) from a CSV with columns: filename, transcript
  - HYP (anonymized transcripts) from per-method LibriSpeech-style .trans.txt files

Expected layout:
  ROOT/
    <method>/<spk>/<chapter>/<uttid>_suffix.wav
    transcripts/
      <method>/trans_txt/<spk>/<chapter>/<spk>-<chapter>.trans.txt

Outputs:
  evaluate_resultsnew/wer_from_transcripts/summary.csv
  evaluate_resultsnew/wer_from_transcripts/details_<method>.csv
"""

import os, re, csv, pathlib
from typing import Dict, Tuple, List, Optional
import pandas as pd
import numpy as np

# --------- EDIT THESE ---------
ROOT = "/home/jacobala@alabsad.fau.de/AOT/Data_governance/anonymized_outputsnew1"
REF_CSV = "/home/jacobala@alabsad.fau.de/AOT/Data_governance/LibriSpeech 4/LibriSpeech-transcript.csv"   # has columns: filename, transcript
# ------------------------------

TRANS_ROOT = os.path.join(ROOT, "transcripts")
OUT_DIR = "evaluate_resultsnew/wer_from_transcripts"
os.makedirs(OUT_DIR, exist_ok=True)

UTTID_RE = re.compile(r"^(\d+-\d+-\d+)")  

# --------- Text norm + WER ----------
def normalize_words(s: str) -> List[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.split()

def word_edit_distance(ref: List[str], hyp: List[str]) -> int:
    n, m = len(ref), len(hyp)
    dp = np.zeros((n+1, m+1), dtype=np.int32)
    for i in range(n+1): dp[i, 0] = i
    for j in range(m+1): dp[0, j] = j
    for i in range(1, n+1):
        ri = ref[i-1]
        for j in range(1, m+1):
            cost = 0 if ri == hyp[j-1] else 1
            dp[i, j] = min(dp[i-1, j]+1, dp[i, j-1]+1, dp[i-1, j-1]+cost)
    return int(dp[n, m])

def corpus_wer(ref_texts: List[str], hyp_texts: List[str]) -> Tuple[float, int, int]:
    total_words = total_edits = 0
    for r, h in zip(ref_texts, hyp_texts):
        rw, hw = normalize_words(r), normalize_words(h)
        total_words += len(rw)
        total_edits += word_edit_distance(rw, hw)
    return 100.0 * total_edits / max(1, total_words), total_words, total_edits

# --------- Load references ----------
def extract_uttid_from_path(p: str) -> Optional[str]:
    base = os.path.splitext(os.path.basename(str(p)))[0]
    m = UTTID_RE.match(base)
    return m.group(1) if m else None

def load_ref_csv(path: str) -> Dict[str, str]:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    # Accept filename/path/file/audio_path/abs_path
    path_col = cols.get("filename") or cols.get("path") or cols.get("file") or cols.get("audio_path") or cols.get("abs_path")
    text_col = cols.get("transcript") or cols.get("text")
    if not path_col or not text_col:
        raise SystemExit(f"REF_CSV needs 'filename' (or path/file) and 'transcript' (or text). Columns: {list(df.columns)}")

    uttids = [extract_uttid_from_path(v) for v in df[path_col].astype(str).tolist()]
    texts  = df[text_col].astype(str).tolist()

    ref_map: Dict[str, str] = {}
    missing = 0
    for u, t in zip(uttids, texts):
        if u: ref_map[u] = t
        else: missing += 1

    if not ref_map:
        raise SystemExit("No valid uttids parsed from REF_CSV filenames.")
    if missing:
        print(f"[WARN] {missing} rows had filenames where uttid could not be parsed and were skipped.")
    print(f"[INFO] Loaded {len(ref_map)} reference transcripts from {path}")
    return ref_map

# --------- Load method hypotheses  ----------
def read_trans_txt(file_path: str) -> Dict[str, str]:
    hyp: Dict[str, str] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(" ", 1)
            if len(parts) != 2: continue
            u, text = parts
            m = UTTID_RE.match(u)
            if m: hyp[m.group(1)] = text.strip()
    return hyp

def load_method_hypotheses(method: str) -> Dict[str, str]:
    root = os.path.join(TRANS_ROOT, method, "trans_txt")
    if not os.path.isdir(root):
        print(f"[WARN] No trans_txt folder for method '{method}': {root}")
        return {}
    hyp_map: Dict[str, str] = {}
    for p in pathlib.Path(root).rglob("*.trans.txt"):
        hyp_map.update(read_trans_txt(str(p)))
    print(f"[INFO] Method '{method}': loaded {len(hyp_map)} uttids from {root}")
    return hyp_map

# --------- Discover methods ----------
def list_methods(root: str) -> List[str]:
    methods: List[str] = []
    for d in sorted(pathlib.Path(root).iterdir()):
        if d.is_dir() and d.name != "transcripts":
            methods.append(d.name)
    return methods

# --------- Main ----------
def main():
    if not os.path.isdir(ROOT):
        raise SystemExit(f"ROOT not found: {ROOT}")
    if not os.path.isfile(REF_CSV):
        raise SystemExit(f"REF_CSV not found: {REF_CSV}")

    ref_map = load_ref_csv(REF_CSV)
    methods = list_methods(ROOT)
    if not methods:
        raise SystemExit(f"No method folders under {ROOT} (except 'transcripts').")

    summaries = []

    for method in methods:
        hyp_map = load_method_hypotheses(method)
        if not hyp_map:
            print(f"[WARN] Skipping '{method}' (no hypotheses).")
            continue

        uttids = sorted(set(ref_map.keys()) & set(hyp_map.keys()))
        if not uttids:
            print(f"[WARN] No overlapping uttids for method '{method}'.")
            continue

        ref_texts = [ref_map[u] for u in uttids]
        hyp_texts = [hyp_map[u] for u in uttids]

        wer, total_words, total_edits = corpus_wer(ref_texts, hyp_texts)

        # Per-utterance details
        detail_rows = []
        for u, r, h in zip(uttids, ref_texts, hyp_texts):
            rw, hw = normalize_words(r), normalize_words(h)
            edits = word_edit_distance(rw, hw)
            wer_u = 100.0 * edits / max(1, len(rw))
            detail_rows.append({
                "uttid": u,
                "ref_text": r,
                "hyp_text": h,
                "ref_words": len(rw),
                "edits": edits,
                "wer_utt_pct": round(wer_u, 2),
            })

        detail_csv = os.path.join(OUT_DIR, f"details_{method}.csv")
        pd.DataFrame(detail_rows).to_csv(detail_csv, index=False, quoting=csv.QUOTE_MINIMAL)

        summaries.append({
            "method": method,
            "n_overlap_utts": len(uttids),
            "total_ref_words": total_words,
            "total_edits": total_edits,
            "wer_pct": round(wer, 3),
            "details_csv": detail_csv,
        })

        print(f"[{method}] WER={wer:.2f}%  overlap_utts={len(uttids)}  words={total_words}  -> {detail_csv}")

    if summaries:
        summary_csv = os.path.join(OUT_DIR, "summary.csv")
        pd.DataFrame(summaries).to_csv(summary_csv, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"\n[WROTE] {summary_csv}")
    else:
        print("\n[INFO] No summaries written — no methods/hypotheses matched your refs.")

if __name__ == "__main__":
    main()
