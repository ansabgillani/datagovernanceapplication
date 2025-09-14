#!/usr/bin/env python3
"""
Plot Clean vs Noisy Speech Waveforms
------------------------------------

This script compares a clean and a noisy speech file
by plotting their waveforms side by side.

Usage:
    python plot_waveforms.py --clean /path/to/clean.wav --noisy /path/to/noisy.wav --output waveform.png

Dependencies:
    - matplotlib
    - librosa
    - numpy
"""

import argparse
import matplotlib.pyplot as plt
import librosa
import numpy as np


def plot_waveforms(clean_file, noisy_file, output_file="clean_vs_noisy_waveform.png"):
    """Plot clean vs noisy waveform and save as image."""

    # Load audio files
    y_clean, sr = librosa.load(clean_file, sr=None)
    y_noisy, _ = librosa.load(noisy_file, sr=None)

    # Time axes
    t_clean = np.linspace(0, len(y_clean) / sr, num=len(y_clean))
    t_noisy = np.linspace(0, len(y_noisy) / sr, num=len(y_noisy))

    # Plot waveforms
    plt.figure(figsize=(12, 5))

    plt.subplot(2, 1, 1)
    plt.plot(t_clean, y_clean, alpha=0.7)
    plt.title("Clean Speech (Waveform)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    plt.plot(t_noisy, y_noisy, alpha=0.7, color="orange")
    plt.title("Noisy Speech (Waveform)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()

    print(f"✅ Saved waveform image as {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot clean vs noisy speech waveforms.")
    parser.add_argument("--clean", "-c", type=str, required=True, help="Path to clean WAV file")
    parser.add_argument("--noisy", "-n", type=str, required=True, help="Path to noisy WAV file")
    parser.add_argument("--output", "-o", type=str, default="clean_vs_noisy_waveform.png", help="Output image file")

    args = parser.parse_args()
    plot_waveforms(args.clean, args.noisy, args.output)
