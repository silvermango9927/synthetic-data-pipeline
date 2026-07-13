"""ASR Scaling Law automated sweeper. Runs training loops on different fractions of data and plots curves."""
import os
import csv
import click
import pandas as pd
from datasets import load_dataset
from benchmark.config import TrainConfig
from benchmark.train import train_asr
from benchmark.evaluate import evaluate_asr

# Medians of clip durations to calculate total hours of subsets
MEDIANS = {
    "zh": 29.5,  # seconds per long clip
    "hi": 34.5   # seconds per long clip
}

def estimate_hours(num_clips: int, lang: str) -> float:
    return (num_clips * MEDIANS.get(lang, 30.0)) / 3600.0

def run_scaling_sweep(
    model_type: str = "whisper",
    model_name: str = "openai/whisper-tiny",
    lang: str = "zh",
    data_type: str = "synthetic",
    dataset_name: str = None,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    use_lora: bool = True,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    device: str = "cuda",
    fractions = [0.1, 0.25, 0.5, 1.0]
):
    """Executes the scaling laws training sweep on target fractions, evaluates each, and logs scaling statistics."""
    print(f"\n====================================================")
    print(f" STARTING ASR SCALING LAW SWEEP ({lang.upper()})")
    print(f" Model: {model_name} | Type: {model_type}")
    print(f" Data Mode: {data_type} | Sweeps: {fractions}")
    print(f"====================================================\n")
    
    # Pre-calculate base dataset size
    if not dataset_name:
        dataset_name = f"silvermango9927/synthetic-asr-{lang}"
        
    # Get total train clips from local cloned repo if available, otherwise fallback to HF Hub
    try:
        from pathlib import Path
        local_repo_dir = Path("outputs/hf_datasets") / f"synthetic-asr-{lang}"
        cfg_name = "long_clean" if data_type in ("synthetic", "both") else "long_augmented"
        
        if local_repo_dir.exists():
            manifest_file = local_repo_dir / "data" / cfg_name / "manifest.jsonl"
            with open(manifest_file, "r", encoding="utf-8") as f:
                total_clips = sum(1 for _ in f)
        else:
            from dotenv import load_dotenv
            load_dotenv()
            token = os.getenv("HF_TOKEN")
            ds_info = load_dataset(dataset_name, cfg_name, split="train", token=token)
            total_clips = len(ds_info)
            
        if data_type == "both":
            total_clips *= 2
    except Exception:
        total_clips = 1000  # Default fallback estimation
        
    results = []
    
    # Save statistics paths
    stats_dir = "outputs/benchmark/stats"
    os.makedirs(stats_dir, exist_ok=True)
    csv_path = os.path.join(stats_dir, f"scaling_{lang}_{model_type}_{data_type}.csv")
    plot_path = os.path.join(stats_dir, f"scaling_{lang}_{model_type}_{data_type}.png")
    
    for f in fractions:
        print(f"\n>>> Running Sweep for Fraction: {f:.1%} <<<")
        sub_clips = int(total_clips * f)
        hours = estimate_hours(sub_clips, lang)
        
        sweep_output_dir = f"outputs/benchmark/scaling_{lang}_{model_type}_{data_type}_f{f}"
        
        # 1. Setup TrainConfig for this fraction run
        cfg = TrainConfig(
            model_name=model_name,
            model_type=model_type,
            lang=lang,
            data_type=data_type,
            fraction=f,
            dataset_name=dataset_name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            use_lora=use_lora,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            device=device,
            output_dir=sweep_output_dir,
            logging_steps=5
        )
        
        # 2. Run Training
        step, train_loss = train_asr(cfg)
        
        # 3. Run Evaluation on validation set
        final_model_path = os.path.join(sweep_output_dir, "final_model")
        wer, cer = evaluate_asr(
            model_path=final_model_path,
            model_type=model_type,
            lang=lang,
            dataset_name=dataset_name,
            data_type=data_type,
            device=device
        )
        
        # 4. Log results
        results.append({
            "Fraction": f,
            "Total_Clips": sub_clips,
            "Hours": hours,
            "Train_Loss": train_loss,
            "WER": wer,
            "CER": cer
        })
        
        # Save statistics table immediately
        with open(csv_path, "w", newline="") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=["Fraction", "Total_Clips", "Hours", "Train_Loss", "WER", "CER"])
            writer.writeheader()
            writer.writerows(results)
            
        print(f"\nScaling results successfully saved/updated to {csv_path}")
        print("\n================== SCALING LAW SUMMARY ==================")
        df = pd.DataFrame(results)
        print(df.to_markdown(index=False))
        print("=========================================================\n")
        
        # Plot scaling curves immediately
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(df["Hours"], df["WER"] * 100, marker='o', linewidth=2, label="WER (%)", color="crimson")
            plt.plot(df["Hours"], df["CER"] * 100, marker='s', linewidth=2, label="CER (%)", color="royalblue")
            
            plt.title(f"VALSEA ASR Scaling Laws - {lang.upper()} ({model_type.upper()} on {data_type.upper()})", fontsize=14, fontweight='bold')
            plt.xlabel("Dataset Size (Hours of Audio)", fontsize=12)
            plt.ylabel("Error Rate (%)", fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(fontsize=11)
            
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Scaling laws visualization curve plotted/updated at {plot_path}")
        except Exception as e:
            print(f"[WARN] Matplotlib plotting failed: {e}")
            
        # 5. Git Commit & Push immediately
        import subprocess
        try:
            subprocess.run(["git", "config", "user.email", "kaggle-worker@gemini.ai"], check=False)
            subprocess.run(["git", "config", "user.name", "Kaggle Worker"], check=False)
            
            subprocess.run(["git", "add", csv_path, plot_path], check=True)
            commit_msg = f"chore: auto-update scaling sweep results for fraction {f:.1%}"
            subprocess.run(["git", "commit", "-m", commit_msg], check=True)
            subprocess.run(["git", "push", "origin", "benchmark/scaling-laws-results"], check=True)
            print(f"🚀 Successfully pushed results for fraction {f:.1%} to GitHub!")
        except Exception as git_err:
            print(f"[WARN] Failed to auto-push results to GitHub: {git_err}")
            
    return results

@click.command()
@click.option("--model", "model_type", type=click.Choice(["whisper", "qwen", "ctc"]), default="whisper")
@click.option("--model-name", default="openai/whisper-tiny")
@click.option("--lang", default="zh", type=click.Choice(["zh", "hi"]))
@click.option("--data-type", default="synthetic", type=click.Choice(["synthetic", "augmented", "both"]))
@click.option("--epochs", default=1, type=int)
@click.option("--batch-size", default=4, type=int)
@click.option("--lr", "learning_rate", default=5e-5, type=float)
@click.option("--use-lora/--no-lora", default=True)
@click.option("--load-in-8bit", is_flag=True, default=False)
@click.option("--load-in-4bit", is_flag=True, default=False)
@click.option("--device", default="cuda")
@click.option("--fractions", default="0.1,0.25,0.5,1.0")
@click.option("--dataset-name", default=None)
def main(fractions: str, **kwargs):
    # Parse list of fractions
    frac_list = [float(f.strip()) for f in fractions.split(",")]
    run_scaling_sweep(fractions=frac_list, **kwargs)

if __name__ == "__main__":
    main()
