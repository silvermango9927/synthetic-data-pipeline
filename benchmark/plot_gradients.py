"""Script to compute and plot the first and second derivatives (gradients) of ASR Scaling Laws."""
import os
import pandas as pd
import numpy as np

def plot_gradients_for_lang(lang: str):
    # Try importing matplotlib inside to handle headless environment if needed
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[ERROR] matplotlib is required to plot gradients.")
        return

    csv_path = f"outputs/benchmark/normalized_stats/scaling_{lang}_whisper_synthetic.csv"
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV path does not exist: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df = df.sort_values("Hours").reset_index(drop=True)

    hours = df["Hours"].values
    wer = df["WER"].values * 100  # Convert to percentage
    cer = df["CER"].values * 100  # Convert to percentage

    # Calculate derivatives using numpy.gradient (which handles non-uniform intervals)
    d1_wer = np.gradient(wer, hours)
    d2_wer = np.gradient(d1_wer, hours)

    d1_cer = np.gradient(cer, hours)
    d2_cer = np.gradient(d1_cer, hours)

    # Setup the multi-panel plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: Original Curves
    axes[0].plot(hours, wer, marker='o', color='crimson', linewidth=2.5, label='WER (%)')
    axes[0].plot(hours, cer, marker='s', color='royalblue', linewidth=2.5, label='CER (%)')
    axes[0].set_title('Original Metrics (Normalized)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Dataset Size (Hours)', fontsize=10)
    axes[0].set_ylabel('Error Rate (%)', fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_ylim(0, 55)
    axes[0].legend(fontsize=10)

    # Panel 2: First Derivative (Gradient)
    axes[1].plot(hours, d1_wer, marker='o', color='darkorange', linewidth=2, linestyle='--', label='d(WER)/d(Hours)')
    axes[1].plot(hours, d1_cer, marker='s', color='teal', linewidth=2, linestyle='--', label='d(CER)/d(Hours)')
    axes[1].axhline(0, color='black', linestyle=':', alpha=0.5)
    axes[1].set_title('1st Derivative (Rate of Improvement)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Dataset Size (Hours)', fontsize=10)
    axes[1].set_ylabel('Change per Hour (% / h)', fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(fontsize=10)

    # Panel 3: Second Derivative (Curvature / Acceleration)
    axes[2].plot(hours, d2_wer, marker='o', color='purple', linewidth=2, linestyle=':', label='d²(WER)/d(Hours)²')
    axes[2].plot(hours, d2_cer, marker='s', color='forestgreen', linewidth=2, linestyle=':', label='d²(CER)/d(Hours)²')
    axes[2].axhline(0, color='black', linestyle=':', alpha=0.5)
    axes[2].set_title('2nd Derivative (Acceleration)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Dataset Size (Hours)', fontsize=10)
    axes[2].set_ylabel('Rate of Change per Hour (% / h²)', fontsize=10)
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].legend(fontsize=10)

    plt.suptitle(f"ASR Scaling Laws Gradient Analysis - {lang.upper()} (Whisper-Tiny)", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_dir = "outputs/benchmark/normalized_stats"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"scaling_{lang}_gradient_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Gradient analysis plot saved to {output_path}")

if __name__ == "__main__":
    for lang in ["zh", "hi"]:
        plot_gradients_for_lang(lang)
