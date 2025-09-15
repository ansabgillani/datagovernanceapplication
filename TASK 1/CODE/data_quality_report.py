#!/usr/bin/env python3
"""
Data Quality Report Generator for Speech Datasets (e.g., LibriSpeech)

Usage:
    python generate_report.py --input librispeech_metadata_all.csv --output data_quality_report
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def main(input_csv, output_dir):
    # ------------------------------
    # Load CSV
    # ------------------------------
    print(f"Loading metadata from {input_csv} ...")
    df = pd.read_csv(input_csv, low_memory=False)

    # Drop duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()]

    # ------------------------------
    # Fix numeric columns
    # ------------------------------
    numeric_cols = ["recording_duration", "sample_rate", "total_dur/spk"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ------------------------------
    # Create output folder
    # ------------------------------
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------
    # 1. Completeness check
    # ------------------------------
    missing = df.isnull().sum()
    missing.to_csv(os.path.join(output_dir, "missing_values.csv"))
    print("Saved missing_values.csv")

    # ------------------------------
    # 2. Recording duration distribution
    # ------------------------------
    if "recording_duration" in df.columns:
        try:
            plt.figure(figsize=(8, 5))
            df["recording_duration"].hist(bins=50)
            plt.title("Distribution of Recording Durations")
            plt.xlabel("Seconds")
            plt.ylabel("Count")
            plt.savefig(os.path.join(output_dir, "duration_distribution.png"))
            plt.close()
            print("Saved duration_distribution.png")
        except Exception as e:
            print("Error duration_distribution:", e)

    # ------------------------------
    # 3. Transcript word count vs duration
    # ------------------------------
    if "text" in df.columns and "recording_duration" in df.columns:
        try:
            df["num_words"] = df["text"].apply(lambda x: len(str(x).split()))
            plt.figure(figsize=(8, 5))
            sns.scatterplot(data=df, x="recording_duration", y="num_words", alpha=0.4)
            plt.title("Consistency: Duration vs Transcript Word Count")
            plt.savefig(os.path.join(output_dir, "duration_vs_words.png"))
            plt.close()
            print("Saved duration_vs_words.png")
        except Exception as e:
            print("Error duration_vs_words:", e)

    # ------------------------------
    # 4. Gender balance
    # ------------------------------
    if "gender" in df.columns:
        try:
            plt.figure(figsize=(6, 4))
            df["gender"].value_counts().plot(kind="bar")
            plt.title("Gender Distribution of Speakers")
            plt.xlabel("Gender")
            plt.ylabel("Count")
            plt.savefig(os.path.join(output_dir, "gender_distribution.png"))
            plt.close()
            print("Saved gender_distribution.png")
        except Exception as e:
            print("Error gender_distribution:", e)

    # ------------------------------
    # 5. Speaker contribution imbalance
    # ------------------------------
    if "total_dur/spk" in df.columns:
        try:
            plt.figure(figsize=(8, 5))
            df["total_dur/spk"].hist(bins=50)
            plt.title("Total Duration per Speaker")
            plt.xlabel("Seconds")
            plt.ylabel("Count")
            plt.savefig(os.path.join(output_dir, "speaker_total_duration.png"))
            plt.close()
            print("Saved speaker_total_duration.png")
        except Exception as e:
            print("Error speaker_total_duration:", e)

    # ------------------------------
    # 6. Language & Country coverage
    # ------------------------------
    if "language" in df.columns:
        try:
            plt.figure(figsize=(8, 5))
            df["language"].value_counts().plot(kind="bar")
            plt.title("Language Distribution")
            plt.savefig(os.path.join(output_dir, "language_distribution.png"))
            plt.close()
            print("Saved language_distribution.png")
        except Exception as e:
            print("Error language_distribution:", e)

    if "country" in df.columns:
        try:
            plt.figure(figsize=(10, 5))
            df["country"].value_counts().head(20).plot(kind="bar")
            plt.title("Top 20 Countries by Recordings")
            plt.savefig(os.path.join(output_dir, "country_distribution.png"))
            plt.close()
            print("Saved country_distribution.png")
        except Exception as e:
            print("Error country_distribution:", e)

    # ------------------------------
    # 7. Outlier detection
    # ------------------------------
    if "recording_duration" in df.columns:
        try:
            q_low, q_high = df["recording_duration"].quantile([0.01, 0.99])
            outliers = df[(df["recording_duration"] < q_low) | (df["recording_duration"] > q_high)]
            outliers.to_csv(os.path.join(output_dir, "outliers.csv"), index=False)
            print(f"Saved outliers.csv ({len(outliers)} rows)")
        except Exception as e:
            print("Error outliers:", e)

    # ------------------------------
    # 8. Summary report
    # ------------------------------
    summary = {
        "Total samples": len(df),
        "Unique speakers": df["speaker_id"].nunique() if "speaker_id" in df.columns else None,
        "Languages": df["language"].nunique() if "language" in df.columns else None,
        "Countries": df["country"].nunique() if "country" in df.columns else None,
        "Missing values": int(missing.sum()),
        "Outlier samples": len(outliers) if "outliers" in locals() else 0,
    }
    summary_df = pd.DataFrame(list(summary.items()), columns=["Metric", "Value"])
    summary_df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    print("Saved summary.csv")

    print(f"\n Data quality report generated in folder: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Data Quality Report for Speech Datasets")
    parser.add_argument("--input", required=True, help="Path to metadata CSV")
    parser.add_argument("--output", default="data_quality_report", help="Output folder for report")
    args = parser.parse_args()
    main(args.input, args.output)
