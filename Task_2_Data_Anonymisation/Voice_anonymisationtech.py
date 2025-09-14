

"""
VOICE ANONYMIZATION PIPELINE (LibriSpeech)

- Recursively scan an input folder for audio files
- Infer speaker as the FIRST directory under the input root (LibriSpeech: <root>/<speaker>/<chapter>/<file>)
- Mirror the entire subfolder tree (speaker/chapter/...) for each output method
- Apply 3 anonymization techniques:
    1) Pitch shift (+n semitones)
    2) Time-stretch (rate)
    3) McAdams SLOW variant -> write header at 11025 Hz (no resampling)

"""

import os, pathlib, argparse
from typing import Iterable, List, Dict
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import lfilter  # for McAdams synth

# -------- inputs --------
DEFAULT_INPUT_DIR = "/home/jacobala@alabsad.fau.de/AOT/Data_governance/LibriSpeech 4/test-clean"
DEFAULT_OUT_DIR   = "/home/jacobala@alabsad.fau.de/AOT/Data_governance/anonymized_outputsnew1"

DEFAULT_PITCH_STEPS = 5
DEFAULT_STRETCH     = 1.5
DEFAULT_EXTS        = (".flac", ".wav", ".mp3", ".ogg", ".m4a")

# McAdams params
DEFAULT_MCADAMS_ALPHA   = 0.8
DEFAULT_MCADAMS_LPCORD  = 20
DEFAULT_MCADAMS_FR_MS   = 30.0
DEFAULT_MCADAMS_HOP_MS  = 10.0

#SLOW header sample-rate (no resampling)
MCADAMS_SLOW_SR = 11025
#MCADAMS_FAST_SR = 21000
# -------------------------


def all_audio_files(root: str, exts: Iterable[str]) -> List[str]:
    root_path = pathlib.Path(root)
    exts_l = tuple(e.lower() for e in exts)
    return [
        str(p)
        for p in root_path.rglob("*")
        if p.is_file() and p.suffix.lower() in exts_l and not p.name.startswith("._")
    ]


def infer_speaker(input_root: str, file_path: str) -> str:
    rel = os.path.relpath(file_path, start=input_root)
    parts = pathlib.Path(rel).parts
    if len(parts) >= 2:
        return parts[0]
    return os.path.basename(os.path.dirname(file_path))


def method_roots(out_dir: str, pitch_steps: int, stretch_rate: float,
                 mcadams_alpha: float) -> Dict[str, str]:
    roots = {
        "pitch":        os.path.join(out_dir, f"pitch_plus{pitch_steps}"),
        "stretch":      os.path.join(out_dir, f"stretch_{stretch_rate}x"),
        # SLOWT variant folder name includes target header sr
        "mcadams_slow": os.path.join(out_dir, f"mcadams_slow_sr{MCADAMS_SLOW_SR}_a{mcadams_alpha:.2f}"),
    }
    for p in roots.values():
        os.makedirs(p, exist_ok=True)
    return roots


def mirror_dest_dir(method_root: str, input_root: str, file_path: str) -> str:
    rel = os.path.relpath(file_path, start=input_root)
    rel_dir = os.path.dirname(rel)  # speaker/chapter/...
    dest_dir = os.path.join(method_root, rel_dir)
    os.makedirs(dest_dir, exist_ok=True)
    return dest_dir


# ------------------ McAdams ------------------
def mcadams_frame(frame: np.ndarray, lpc_order: int, alpha: float) -> np.ndarray:
    if len(frame) <= lpc_order + 2 or np.allclose(frame, 0.0):
        return frame.copy()

    try:
        a = librosa.lpc(frame, order=lpc_order)  
    except Exception:
        return frame.copy()

    res = lfilter(a, [1.0], frame)  

    roots = np.roots(a)
    new_roots = []
    for r in roots:
        if np.iscomplex(r) and abs(r.imag) > 1e-12:
            rho = np.abs(r)
            phi = np.angle(r)
            new_phi = np.sign(phi) * (abs(phi) ** alpha)
            r2 = rho * np.exp(1j * new_phi)
            mag = np.abs(r2)
            if mag >= 0.9999:
                r2 = r2 / mag * 0.9999
            new_roots.append(r2)
        else:
            new_roots.append(r)

    a_mod = np.poly(new_roots).real
    y = lfilter([1.0], a_mod, res)  
    return y


def mcadams_anonymize(y: np.ndarray, sr: int, alpha: float,
                      lpc_order: int = 20, frame_ms: float = 30.0, hop_ms: float = 10.0) -> np.ndarray:
    frame_len = max(32, int(round(sr * frame_ms / 1000.0)))
    hop_len   = max(16, int(round(sr * hop_ms / 1000.0)))
    win = np.hanning(frame_len).astype(np.float32)

    n = len(y)
    out = np.zeros(n + frame_len, dtype=np.float32)
    norm = np.zeros_like(out)

    for start in range(0, n, hop_len):
        end = start + frame_len
        frame = np.zeros(frame_len, dtype=np.float32)
        chunk = y[start:min(end, n)]
        frame[:len(chunk)] = chunk

        yf = mcadams_frame(frame, lpc_order=lpc_order, alpha=alpha)
        yf_win = yf * win
        out[start:start + frame_len] += yf_win
        norm[start:start + frame_len] += win ** 2

    nz = norm > 1e-8
    out[nz] /= norm[nz]
    out = out[:n]

    peak = np.max(np.abs(out)) + 1e-9
    return (out / peak).astype(np.float32)
# --------------------------------------------------


def process_batch(
    input_dir: str,
    out_dir: str,
    pitch_steps: int,
    stretch_rate: float,
    exts: Iterable[str],
    mcadams_alpha: float,
    mcadams_lpc_order: int,
    mcadams_frame_ms: float,
    mcadams_hop_ms: float,
):
    files = all_audio_files(input_dir, exts)
    if not files:
        print("[WARN] No audio files found. Check your --input path and extensions.")
        return

    roots = method_roots(out_dir, pitch_steps, stretch_rate, mcadams_alpha)

    for i, fpath in enumerate(sorted(files), start=1):
        base_noext = os.path.splitext(os.path.basename(fpath))[0]

        try:
            y, sr = librosa.load(fpath, sr=None, mono=True)
        except Exception as e:
            print(f"[SKIP] {fpath} (read error: {e})")
            continue

        # ---- 1) Pitch shift (keeps original sr)
        try:
            y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_steps)
            ddir = mirror_dest_dir(roots["pitch"], input_dir, fpath)
            sf.write(os.path.join(ddir, f"{base_noext}_pitch+{pitch_steps}.wav"),
                     y_pitch, int(sr), subtype="PCM_16")
        except Exception as e:
            print(f"[ERR] Pitch shift failed for {fpath}: {e}")

        # ---- 2) Time stretch (changes duration by design, keeps sr)
        try:
            if stretch_rate <= 0:
                raise ValueError("stretch_rate must be > 0")
            y_stretch = librosa.effects.time_stretch(y, rate=stretch_rate)
            ddir = mirror_dest_dir(roots["stretch"], input_dir, fpath)
            sf.write(os.path.join(ddir, f"{base_noext}_stretch{stretch_rate}x.wav"),
                     y_stretch, int(sr), subtype="PCM_16")
        except Exception as e:
            print(f"[ERR] Time-stretch failed for {fpath}: {e}")

        # ---- 3) McAdams SLOW variant: write header at 11 kHz (no resampling)
        try:
            y_mcad = mcadams_anonymize(
                y=y.astype(np.float32),
                sr=sr,
                alpha=mcadams_alpha,
                lpc_order=mcadams_lpc_order,
                frame_ms=mcadams_frame_ms,
                hop_ms=mcadams_hop_ms,
            )
            ddir = mirror_dest_dir(roots["mcadams_slow"], input_dir, fpath)
            # IMPORTANT: write with header samplerate = 21,000 Hz
            sf.write(os.path.join(ddir, f"{base_noext}_mcadams_slow_sr{MCADAMS_SLOW_SR}_a{mcadams_alpha:.2f}.wav"),
                     y_mcad, int(MCADAMS_SLOW_SR), subtype="PCM_16")
        except Exception as e:
            print(f"[ERR] McAdams slow failed for {fpath}: {e}")

        if i % 50 == 0 or i == len(files):
            print(f"[PROGRESS] {i}/{len(files)} files processed")

    print("\n[DONE] Outputs:")
    for k, v in roots.items():
        print(f"  - {k:12s}: {v}")


def parse_args():
    p = argparse.ArgumentParser(description="Voice anonymization pipeline (LibriSpeech).")
    p.add_argument("--input", type=str, default=DEFAULT_INPUT_DIR,
                   help="Input root (e.g., .../LibriSpeech/test-clean)")
    p.add_argument("--out", type=str, default=DEFAULT_OUT_DIR,
                   help="Output root folder.")
    p.add_argument("--pitch_steps", type=int, default=DEFAULT_PITCH_STEPS)
    p.add_argument("--stretch_rate", type=float, default=DEFAULT_STRETCH)
    p.add_argument("--exts", type=str, nargs="+", default=list(DEFAULT_EXTS),
                   help="Extensions to include, e.g. .flac .wav .mp3 .ogg .m4a")
    # McAdams knobs
    p.add_argument("--mcadams_alpha", type=float, default=DEFAULT_MCADAMS_ALPHA,
                   help="McAdams coefficient (e.g., 0.8).")
    p.add_argument("--mcadams_lpc_order", type=int, default=DEFAULT_MCADAMS_LPCORD,
                   help="LPC order (e.g., 20).")
    p.add_argument("--mcadams_frame_ms", type=float, default=DEFAULT_MCADAMS_FR_MS,
                   help="McAdams frame length in ms (e.g., 30).")
    p.add_argument("--mcadams_hop_ms", type=float, default=DEFAULT_MCADAMS_HOP_MS,
                   help="McAdams hop length in ms (e.g., 10).")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.input):
        raise FileNotFoundError(f"Input folder not found: {args.input}")
    os.makedirs(args.out, exist_ok=True)

    process_batch(
        input_dir=args.input,
        out_dir=args.out,
        pitch_steps=args.pitch_steps,
        stretch_rate=args.stretch_rate,
        exts=args.exts,
        mcadams_alpha=args.mcadams_alpha,
        mcadams_lpc_order=args.mcadams_lpc_order,
        mcadams_frame_ms=args.mcadams_frame_ms,
        mcadams_hop_ms=args.mcadams_hop_ms,
    )


if __name__ == "__main__":
    main()
