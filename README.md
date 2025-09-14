
# Voice Anonymization Pipeline

This project provides a **complete pipeline** to anonymize speech, transcribe it, and evaluate the privacy–utility trade-offusing EER (privacy) and WER (utility)on LibriSpeech datasets.

---

## Scripts

- **Voiceanonymisationtech.py**  
  Generates anonymized versions of speech with three methods:
  1. Pitch shift (e.g., +5 semitones)  
  2. Time-stretch (e.g., 1.5× speed change)  
  3. McAdams (SLOW, writes header at 11025 Hz without resampling)

- **`EER_eval.py`**  
  Evaluates **privacy** with a speaker-verification model (SpeechBrain ECAPA).  
  Produces baseline (orig↔orig) EER/AUC and per-method (orig↔anon) EER with %SAME at baseline threshold.

- **`whispertranscript.py`**  
  Uses **OpenAI Whisper** to transcribe anonymized audio.  
  Outputs per-file `.txt`, per-chapter `.trans.txt`, per-method CSV, and a summary CSV.

- **`WERanonymized.py`**  
  Evaluates **utility** by computing WER between reference transcripts and anonymized transcripts.

---

## Installation

> Requires Python 3.9–3.11. GPU is optional but recommended.

```bash
# Core packages
pip install numpy scipy pandas librosa soundfile

# PyTorch + Torchaudio
pip install torch torchaudio   # (CPU)
# For GPU, install the correct CUDA build from https://pytorch.org

# Speech / ASR
pip install speechbrain openai-whisper

