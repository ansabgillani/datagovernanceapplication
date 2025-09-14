
"""


Use OpenAI Whisperto transcribe ALL anonymized audios per method and write:
  1) per-file TXT:   <TRANSCRIPT_ROOT>/<method>/txt/<rel_path>.txt
  2) per-chapter:    <TRANSCRIPT_ROOT>/<method>/trans_txt/<spk>/<chapter>/<spk>-<chapter>.trans.txt
  3) per-method CSV: <TRANSCRIPT_ROOT>/transcripts_<method>.csv
  4) summary CSV:    <TRANSCRIPT_ROOT>/_summary.csv

Assumes LibriSpeech-like layout:
  <ROOT>/<method>/<spk>/<chapter>/<uttid>_anything.wav

"""

import os
import re
import csv
import pathlib
from typing import List, Dict, Optional, Tuple

import torch
import torchaudio
import pandas as pd
import whisper  # OpenAI Whisper (pip package: openai-whisper)

# ===================== CONFIG=====================
ROOT = "/home/jacobala@alabsad.fau.de/AOT/Data_governance/anonymized_outputsnew1"
TRANSCRIPT_ROOT = os.path.join(ROOT, "transcripts")

# Auto-detect method subfolders under ROOT (e.g., mcadams_slow_sr11025_a0.80, pitch_plus5, stretch_1.5x, ...)
METHOD_DIRS: Optional[Dict[str, str]] = None  # leave None to auto-detect; or provide dict {"method": f"{ROOT}/method"}

# Whisper settings
MODEL_SIZE = "small"     
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP16 = DEVICE == "cuda"

WHISPER_ARGS = dict(
    language="en",                   # set None to auto-detect
    task="transcribe",
    verbose=False,
    temperature=0.0,                 
    condition_on_previous_text=False 
)
# Audio & I/O
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")
MIN_DURATION_SEC = 0.05  # skip absurdly short files
# ================================================================


def list_methods(root: str) -> Dict[str, str]:
    """Auto-discover method subfolders (name -> absolute path)."""
    methods: Dict[str, str] = {}
    p = pathlib.Path(root)
    if not p.exists():
        return methods
    for q in sorted(p.iterdir()):
        if q.is_dir():
            methods[q.name] = str(q)
    return methods


def list_audio(root: str) -> List[str]:
    """Recursively list audio files under root."""
    files: List[str] = []
    for p in pathlib.Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS and not p.name.startswith("._"):
            files.append(str(p))
    return sorted(files)


def rel_speaker_chapter(rel_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Expect LibriSpeech-like structure: <spk>/<chapter>/<file>
    Returns (spk, chapter) or (None, None) if not available.
    """
    parts = rel_path.replace("\\", "/").split("/")
    if len(parts) >= 3:
        return parts[0], parts[1]
    return None, None


def uttid_from_basename(basename: str) -> Optional[str]:
    """
    Recover utt-id like 61-70968-0021 from anonymized basenames such as:
      61-70968-0021_mcadams_slow_sr11025_a0.80.wav -> 61-70968-0021
    """
    m = re.match(r"^(\d+-\d+-\d+)", basename)
    return m.group(1) if m else None


def safe_duration(path: str) -> Optional[float]:
    """Duration in seconds using torchaudio.info; returns None if unavailable."""
    try:
        info = torchaudio.info(path)
        sr = info.sample_rate
        return info.num_frames / float(sr) if sr > 0 else None
    except Exception:
        return None


def load_whisper_model() -> "whisper.Whisper":
    print(f"[INFO] Loading Whisper model: {MODEL_SIZE} on {DEVICE} (fp16={FP16})")
    model = whisper.load_model(MODEL_SIZE, device=DEVICE)
    return model


def transcribe_one(model: "whisper.Whisper", path: str) -> str:
    """
    Run Whisper on one file; returns a single-line transcript.
    """
    try:
        result = model.transcribe(path, fp16=FP16, **WHISPER_ARGS)
        text = (result.get("text") or "").strip()
        return text
    except TypeError:
        # Older whisper versions may not accept all args; fall back
        result = model.transcribe(path)
        return (result.get("text") or "").strip()


def main():
    if not os.path.isdir(ROOT):
        raise SystemExit(f"ROOT not found: {ROOT}")

    methods = METHOD_DIRS if METHOD_DIRS is not None else list_methods(ROOT)
    if not methods:
        raise SystemExit(f"No method subfolders found under: {ROOT}")

    os.makedirs(TRANSCRIPT_ROOT, exist_ok=True)
    model = load_whisper_model()

    summary_rows = []

    for method, folder in methods.items():
        if not os.path.isdir(folder):
            print(f"[WARN] Skipping '{method}' — not a folder: {folder}")
            continue

        print(f"\n=== Transcribing method: {method} ===")
        files = list_audio(folder)
        if not files:
            print(f"[INFO] No audio files in: {folder}")
            continue

        # Outputs
        out_txt_root   = os.path.join(TRANSCRIPT_ROOT, method, "txt")
        out_trans_root = os.path.join(TRANSCRIPT_ROOT, method, "trans_txt")
        os.makedirs(out_txt_root, exist_ok=True)
        os.makedirs(out_trans_root, exist_ok=True)

        rows = []
        # Accumulate chapter lines: (spk, chapter) -> list[(uttid, text)]
        chapter_lines: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}

        for i, apath in enumerate(files, 1):
            rel  = os.path.relpath(apath, start=folder)
            base = os.path.splitext(os.path.basename(apath))[0]
            spk, chap = rel_speaker_chapter(rel)
            uttid = uttid_from_basename(base)
            dur = safe_duration(apath)

            if dur is not None and dur < MIN_DURATION_SEC:
                print(f"[{i}/{len(files)}] {rel} -> SKIP too short ({dur:.3f}s)")
                rows.append({
                    "method": method, "rel_path": rel, "abs_path": apath,
                    "speaker": spk, "chapter": chap, "uttid": uttid,
                    "duration_sec": round(dur, 3), "txt_path": "", "trans_txt_path": "",
                    "transcript": "", "error": f"too_short({dur:.3f}s)"
                })
                continue

            print(f"[{i}/{len(files)}] {rel}")
            hyp, err = "", ""
            try:
                hyp = transcribe_one(model, apath)
            except Exception as e:
                err = f"transcribe_error: {e}"

            # 1) per-file TXT (mirror rel path under txt/)
            txt_out_dir = os.path.join(out_txt_root, os.path.dirname(rel))
            os.makedirs(txt_out_dir, exist_ok=True)
            txt_path = os.path.join(txt_out_dir, f"{base}.txt")
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(hyp)
            except Exception as e:
                err = f"{err}; write_txt_error: {e}" if err else f"write_txt_error: {e}"
                txt_path = ""

            # 2) collect chapter line: "uttid TEXT"
            if spk and chap and uttid and hyp:
                chapter_lines.setdefault((spk, chap), []).append((uttid, hyp))

            rows.append({
                "method": method, "rel_path": rel, "abs_path": apath,
                "speaker": spk, "chapter": chap, "uttid": uttid,
                "duration_sec": round(dur, 3) if dur is not None else None,
                "txt_path": txt_path, "transcript": hyp, "error": err,
                "trans_txt_path": ""  # filled after writing .trans.txt files
            })

        # 3) write per-chapter LibriSpeech-style <spk>-<chapter>.trans.txt
        for (spk, chap), pairs in chapter_lines.items():
            pairs_sorted = sorted(pairs, key=lambda x: x[0])  # sort by uttid
            chapter_dir = os.path.join(out_trans_root, spk, chap)
            os.makedirs(chapter_dir, exist_ok=True)
            trans_txt_path = os.path.join(chapter_dir, f"{spk}-{chap}.trans.txt")
            with open(trans_txt_path, "w", encoding="utf-8") as f:
                for u, text in pairs_sorted:
                    f.write(f"{u} {text}\n")
            # attach path back to matching rows
            for r in rows:
                if r["speaker"] == spk and r["chapter"] == chap:
                    r["trans_txt_path"] = trans_txt_path

        # 4) per-method CSV
        csv_path = os.path.join(TRANSCRIPT_ROOT, f"transcripts_{method}.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
        n_ok = sum(1 for r in rows if not r["error"])
        n_err = len(rows) - n_ok
        print(f"[WROTE] {csv_path}  ({n_ok} ok, {n_err} with errors)")

        summary_rows.append({
            "method": method,
            "folder": folder,
            "n_files": len(files),
            "n_transcribed_ok": n_ok,
            "n_errors": n_err,
            "csv_path": csv_path,
            "trans_txt_root": os.path.join(out_trans_root),
        })

    # 5) summary CSV
    summary_csv = os.path.join(TRANSCRIPT_ROOT, "_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    print("\n[WROTE] summary:", summary_csv)
    for r in summary_rows:
        print(" -", r["method"], "->", r["csv_path"])


if __name__ == "__main__":
    main()
