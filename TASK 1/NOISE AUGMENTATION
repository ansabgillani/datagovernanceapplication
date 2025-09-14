#!/usr/bin/env python3
"""
LibriSpeech Noise Augmentation Script
-------------------------------------

This script walks through the nested LibriSpeech directory structure,
adds Gaussian noise to each WAV file, and saves the noisy version
in a mirrored output directory.

Usage:
    python add_noise_librispeech.py --input /path/to/LibriSpeech --output /path/to/noisy_output --noise 0.005

Dependencies:
    - librosa
    - soundfile
    - numpy

Install with:
    pip install librosa soundfile numpy
"""

import os
import librosa
import soundfile as sf
import numpy as np
import argparse


def add_noise(y, noise_factor=0.005):
    """Add Gaussian noise to an audio signal."""
    noise = np.random.normal(0, noise_factor, y.shape)
    return y + noise


def process_librispeech(input_dir, output_dir, noise_factor=0.005, sr=16000):
    """Walk through LibriSpeech and create noisy copies of all WAV files."""
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                input_path = os.path.join(root, file)

                # Create mirrored output path
                rel_path = os.path.relpath(root, input_dir)
                output_folder = os.path.join(output_dir, rel_path)
                os.makedirs(output_folder, exist_ok=True)
                output_path = os.path.join(output_folder, file)

                # Load audio
                y, sr = librosa.load(input_path, sr=sr)

                # Add noise
                y_noisy = add_noise(y, noise_factor=noise_factor)

                # Save noisy file
                sf.write(output_path, y_noisy, sr)

    print(f"✅ All noisy files saved in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add Gaussian noise to LibriSpeech audio files.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to LibriSpeech root directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to save noisy dataset")
    parser.add_argument("--noise", "-n", type=float, default=0.005, help="Noise factor (default=0.005)")
    parser.add_argument("--sr", type=int, default=16000, help="Target sampling rate (default=16000)")

    args = parser.parse_args()

    process_librispeech(args.input, args.output, args.noise, args.sr)
