
"""
Baseline (orig↔orig) + per-method (orig↔anon) EER/AUC,
and %SAME at baseline thr@EER for calibrated-attacker analysis.

Outputs:
- evaluate_results/orig_orig_eer_summary.csv
- evaluate_results/eer_summary.csv  (with %SAME at baseline thr)


"""

import os, glob, random, pathlib, csv
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from speechbrain.inference.speaker import SpeakerRecognition


INPUT_ORIG = "/home/jacobala@alabsad.fau.de/AOT/Data_governance/LibriSpeech 4/test-clean"
OUT_DIR    = "/home/jacobala@alabsad.fau.de/AOT/Data_governance/anonymized_outputsnew1"
N_IMPOSTOR_PER_METHOD = 4000   # impostor pairs per method (orig↔anon)
BASELINE_MAX_GENUINE_PER_SPK = 100  # cap genuine pairs per speaker (orig↔orig)
BASELINE_N_IMPOSTOR = 4000          # total impostor pairs (orig↔orig)
RANDOM_SEED = 42
AUDIO_EXTS = {".flac", ".wav", ".mp3", ".ogg", ".m4a"}
# -------------------------------


def list_audio(root: str) -> List[str]:
    paths = []
    for p in pathlib.Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS and not p.name.startswith("._"):
            paths.append(str(p))
    return sorted(paths)

def get_methods(out_dir: str) -> Dict[str, str]:
    methods = {}
    for p in sorted(pathlib.Path(out_dir).iterdir()):
        if p.is_dir():
            methods[p.name] = str(p)
    return methods

def find_matching_anonymized(method_root: str, input_root: str, orig_path: str) -> Optional[str]:
    
    rel = os.path.relpath(orig_path, start=input_root)
    rel_dir = os.path.dirname(rel)
    base = os.path.splitext(os.path.basename(rel))[0]
    cand_dir = os.path.join(method_root, rel_dir)
    if not os.path.isdir(cand_dir):
        return None
    cands = []
    for ext in AUDIO_EXTS:
        cands.extend(glob.glob(os.path.join(cand_dir, f"{base}_*{ext}")))
    if not cands:
        return None
    cands.sort(key=os.path.getmtime)
    return cands[-1]

def speaker_from_relpath(rel_path: str) -> Optional[str]:
    # LibriSpeech/test-clean/<spk>/<chapter>/<utt>.<ext> -> spk
    parts = rel_path.replace("\\", "/").split("/")
    if len(parts) >= 3:
        return parts[0]
    return None

def load_verifier(device: Optional[str] = None) -> SpeakerRecognition:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading SpeechBrain ECAPA on {device} ...")
    ver = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )
    return ver

@torch.no_grad()
def cosine_filepair(ver: SpeakerRecognition, f1: str, f2: str) -> float:
    score, _ = ver.verify_files(f1, f2)  # cosine in [-1,1]
    return float(score.cpu().numpy()) if isinstance(score, torch.Tensor) else float(score)

def compute_eer_thr_auc(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float, float]:
    """Compute EER, thr@EER, and AUC (0..1) with pure NumPy."""
    labels = labels.astype(int)
    scores = scores.astype(float)
    order = np.argsort(-scores)
    s = scores[order]
    y = labels[order]
    P = np.sum(y == 1)
    N = np.sum(y == 0)
    if P == 0 or N == 0:
        raise ValueError("Need both genuine (1) and impostor (0) samples.")
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    tpr = tp / P
    fpr = fp / N
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fnr - fpr)))
    eer = float((fnr[idx] + fpr[idx]) / 2.0)
    thr_at_eer = float(s[idx])
    order_fpr = np.argsort(fpr)
    auc = float(np.trapz(y=tpr[order_fpr], x=fpr[order_fpr]))
    return float(np.clip(eer, 0.0, 1.0)), thr_at_eer, float(np.clip(auc, 0.0, 1.0))

# ----------------------- baseline orig↔orig --------------------
def eval_baseline_orig_orig(ver: SpeakerRecognition, orig_by_spk: Dict[str, List[str]]) -> dict:
    rng = random.Random(RANDOM_SEED)

    # Genuine: pair consecutive files per speaker (capped)
    gen_scores: List[float] = []
    for spk, files in orig_by_spk.items():
        if len(files) < 2:
            continue
        files_sorted = sorted(files)
        pairs = []
        for i in range(len(files_sorted) - 1):
            pairs.append((files_sorted[i], files_sorted[i+1]))
            if len(pairs) >= BASELINE_MAX_GENUINE_PER_SPK:
                break
        for f1, f2 in pairs:
            try:
                gen_scores.append(cosine_filepair(ver, f1, f2))
            except Exception:
                continue

    # Impostor: random cross-speaker pairs
    speakers = [s for s,f in orig_by_spk.items() if len(f) > 0]
    imp_scores: List[float] = []
    trials = 0
    max_trials = BASELINE_N_IMPOSTOR * 4
    while len(imp_scores) < BASELINE_N_IMPOSTOR and trials < max_trials and len(speakers) >= 2:
        sa, sb = rng.sample(speakers, 2)
        o1 = rng.choice(orig_by_spk[sa])
        o2 = rng.choice(orig_by_spk[sb])
        trials += 1
        try:
            imp_scores.append(cosine_filepair(ver, o1, o2))
        except Exception:
            continue

    if not gen_scores or not imp_scores:
        raise RuntimeError("Baseline: insufficient pairs scored.")

    labels = np.concatenate([np.ones(len(gen_scores), int),
                             np.zeros(len(imp_scores), int)])
    scores = np.concatenate([np.asarray(gen_scores, float),
                             np.asarray(imp_scores, float)])
    eer, thr, auc = compute_eer_thr_auc(labels, scores)

    print("\n[BASELINE] orig↔orig")
    print(f"  Genuine: {len(gen_scores):5d} | Impostor: {len(imp_scores):5d}")
    print(f"  EER: {eer*100:.3f}% | thr@EER: {thr:.6f} | AUC: {auc:.6f}")

    return {
        "protocol": "orig-orig",
        "n_genuine": len(gen_scores),
        "n_impostor": len(imp_scores),
        "eer_pct": round(100.0 * eer, 3),
        "thr_at_eer": round(thr, 6),
        "auc": round(auc, 6),
        "genuine_mean": round(float(np.mean(gen_scores)), 6),
        "impostor_mean": round(float(np.mean(imp_scores)), 6),
    }

# ------------------------ per-method eval ----------------------
def eval_methods_orig_anon(ver: SpeakerRecognition, orig_by_spk: Dict[str, List[str]], baseline_thr: Optional[float]) -> List[dict]:
    methods = get_methods(OUT_DIR)
    if not methods:
        print(f"[INFO] No method subfolders in: {OUT_DIR}")
        return []
    print("[INFO] Methods detected:")
    for m in methods:
        print(f"  - {m}")

    rng = random.Random(RANDOM_SEED)
    rows = []

    for method, mroot in methods.items():
        print(f"\n[METHOD] {method}")
        gen_scores: List[float] = []
        imp_scores: List[float] = []
        anon_by_spk: Dict[str, List[str]] = {}

        # Genuine: original vs its anonymized counterpart (same spk)
        for spk, origs in orig_by_spk.items():
            for op in origs:
                anon = find_matching_anonymized(mroot, INPUT_ORIG, op)
                if not anon or not os.path.isfile(anon):
                    continue
                anon_by_spk.setdefault(spk, []).append(anon)
                try:
                    gen_scores.append(cosine_filepair(ver, op, anon))
                except Exception:
                    continue

        if not gen_scores:
            print(f"[WARN] No genuine pairs for {method}. Skipping.")
            continue

        # Impostor: original A vs anonymized B (A!=B)
        spks = [s for s in anon_by_spk.keys() if s in orig_by_spk]
        if len(spks) < 2:
            print(f"[WARN] Not enough speakers for impostors in {method}. Skipping.")
            continue

        trials = 0
        max_trials = N_IMPOSTOR_PER_METHOD * 4
        while len(imp_scores) < N_IMPOSTOR_PER_METHOD and trials < max_trials:
            sa, sb = rng.sample(spks, 2)
            o = rng.choice(orig_by_spk[sa])
            a = rng.choice(anon_by_spk[sb])
            trials += 1
            try:
                imp_scores.append(cosine_filepair(ver, o, a))
            except Exception:
                continue

        labels = np.concatenate([np.ones(len(gen_scores), int),
                                 np.zeros(len(imp_scores), int)])
        scores = np.concatenate([np.asarray(gen_scores, float),
                                 np.asarray(imp_scores, float)])
        eer, thr, auc = compute_eer_thr_auc(labels, scores)

        # %SAME at baseline threshold 
        pct_same_gen = pct_same_imp = None
        if baseline_thr is not None:
            gs = np.asarray(gen_scores, float)
            is_ = np.asarray(imp_scores, float)
            pct_same_gen = 100.0 * (gs >= baseline_thr).mean()
            pct_same_imp = 100.0 * (is_ >= baseline_thr).mean()
            print(f"  %SAME@baseline_thr={baseline_thr:.6f}  genuine={pct_same_gen:.2f}%  impostor={pct_same_imp:.2f}%")

        print(f"  Genuine: {len(gen_scores):5d} | Impostor: {len(imp_scores):5d}")
        print(f"  EER: {eer*100:.3f}% | thr@EER: {thr:.6f} | AUC: {auc:.6f}")

        row = {
            "method": method,
            "n_genuine": len(gen_scores),
            "n_impostor": len(imp_scores),
            "eer_pct": round(100.0 * eer, 3),
            "thr_at_eer": round(thr, 6),
            "auc": round(auc, 6),
            "genuine_mean": round(float(np.mean(gen_scores)), 6),
            "impostor_mean": round(float(np.mean(imp_scores)), 6),
        }
        if baseline_thr is not None:
            row["baseline_thr_used"] = round(float(baseline_thr), 6)
            row["pct_same_gen_at_baseline_thr"] = round(float(pct_same_gen), 3)
            row["pct_same_imp_at_baseline_thr"] = round(float(pct_same_imp), 3)

        rows.append(row)

    return rows

# ---------------------------- main ----------------------------
def main():
    if not os.path.isdir(INPUT_ORIG):
        raise SystemExit(f"Input originals not found: {INPUT_ORIG}")
    if not os.path.isdir(OUT_DIR):
        print(f"[WARN] Out dir not found (skipping per-method eval): {OUT_DIR}")

    print("[INFO] Indexing originals ...")
    originals = list_audio(INPUT_ORIG)
    if not originals:
        raise SystemExit(f"No audio found under: {INPUT_ORIG}")

    orig_by_spk: Dict[str, List[str]] = {}
    for op in originals:
        rel = os.path.relpath(op, start=INPUT_ORIG)
        spk = speaker_from_relpath(rel)
        if spk:
            orig_by_spk.setdefault(spk, []).append(op)

    ver = load_verifier()
    os.makedirs("evaluate_resultsnew", exist_ok=True)

    # Baseline
    baseline_row = eval_baseline_orig_orig(ver, orig_by_spk)
    baseline_thr = float(baseline_row["thr_at_eer"])
    with open(os.path.join("evaluate_resultsnew", "orig_orig_eer_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(baseline_row.keys()))
        w.writeheader()
        w.writerow(baseline_row)
    print("[WROTE] evaluate_resultsnew/orig_orig_eer_summary.csv")

    # Per-method (orig↔anon) with %SAME at baseline thr
    if os.path.isdir(OUT_DIR):
        method_rows = eval_methods_orig_anon(ver, orig_by_spk, baseline_thr)
        if method_rows:
            out_csv = os.path.join("evaluate_resultsnew", "eer_summary.csv")
            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(method_rows[0].keys()))
                w.writeheader()
                w.writerows(method_rows)
            print("[WROTE] evaluate_resultsnew/eer_summary.csv")
        else:
            print("[INFO] No per-method results written.")

if __name__ == "__main__":
    main()
