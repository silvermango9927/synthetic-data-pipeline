"""Script to replot ASR scaling curves directly from CSV files to adjust Y-axis limits."""
import os
import pandas as pd
import matplotlib.pyplot as plt
from benchmark.plot_gradients import plot_gradients_for_lang

def replot_scaling_curves(lang: str):
    csv_path = f"outputs/benchmark/normalized_stats/scaling_{lang}_whisper_synthetic.csv"
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV path does not exist: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df = df.sort_values("Hours").reset_index(drop=True)

    plt.figure(figsize=(10, 6))
    plt.plot(df["Hours"], df["WER"] * 100, marker='o', linewidth=2, label="WER (%)", color="crimson")
    plt.plot(df["Hours"], df["CER"] * 100, marker='s', linewidth=2, label="CER (%)", color="royalblue")
    
    plt.title(f"VALSEA ASR Scaling Laws (Normalized) - {lang.upper()} (WHISPER)", fontsize=14, fontweight='bold')
    plt.xlabel("Dataset Size (Hours of Audio)", fontsize=12)
    plt.ylabel("Error Rate (%)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(0, 55)
    plt.legend(fontsize=11)
    
    output_dir = "outputs/benchmark/normalized_stats"
    plot_path = os.path.join(output_dir, f"scaling_{lang}_whisper_synthetic.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Re-plotted scaling curve saved to {plot_path}")

if __name__ == "__main__":
    for lang in ["zh", "hi"]:
        replot_scaling_curves(lang)
        plot_gradients_for_lang(lang)
