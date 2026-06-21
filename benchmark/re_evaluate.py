"""Script to re-evaluate existing ASR checkpoints with text normalization and word segmentation."""
import os
import re
import csv
import click
import torch
import jiwer
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from transformers import AutoProcessor, WhisperForConditionalGeneration

# Medians of clip durations to calculate total hours of subsets
MEDIANS = {
    "zh": 29.5,
    "hi": 34.5
}

def estimate_hours(num_clips: int, lang: str) -> float:
    return (num_clips * MEDIANS.get(lang, 30.0)) / 3600.0

def normalize_text(text: str, lang: str) -> str:
    """Applies standard normalization, removes punctuation, and character-tokenizes Chinese."""
    if not text:
        return ""
    
    text = text.lower().strip()
    
    # Remove common punctuation (both English and Unicode/Chinese punctuation)
    punctuation_regex = r'[.,\/#!$%\^&\*;:{}=\-_`~()?"\'，。！？：、（）《》“”|]'
    text = re.sub(punctuation_regex, '', text)
    
    if lang == "zh":
        # Insert spaces around all Chinese characters (range \u4e00-\u9fff) to allow character-based WER
        text = re.sub(r'([\u4e00-\u9fff])', r' \1 ', text)
        
    # Collapse multiple whitespaces
    text = " ".join(text.split())
    return text

def re_evaluate_sweeps(lang: str, model_type: str = "whisper", data_type: str = "synthetic", device: str = "cuda"):
    print(f"\n====================================================")
    print(f" STARTING RE-EVALUATION WITH NORMALIZATION ({lang.upper()})")
    print(f"====================================================\n")
    
    device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Find all existing checkpoint folders for this sweep
    base_dir = Path("outputs/benchmark")
    pattern = f"scaling_{lang}_{model_type}_{data_type}_f*"
    
    sweep_dirs = sorted(
        list(base_dir.glob(pattern)),
        key=lambda p: float(p.name.split("_f")[-1])
    )
    
    if not sweep_dirs:
        print(f"[ERROR] No folders found matching pattern: {base_dir / pattern}")
        return
        
    print(f"Found {len(sweep_dirs)} sweeps to re-evaluate:")
    for sd in sweep_dirs:
        print(f"  - {sd.name}")
        
    results = []
    
    # Setup new output folder
    output_dir = Path("outputs/benchmark/normalized_stats")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for sd in sweep_dirs:
        fraction_str = sd.name.split("_f")[-1]
        fraction = float(fraction_str)
        
        final_model_path = sd / "final_model"
        if not final_model_path.exists():
            print(f"[WARN] Skipping {sd.name} - 'final_model' not found.")
            continue
            
        print(f"\n>>> Re-evaluating fraction {fraction_str} ({fraction:.1%}) <<<")
        
        # Load processor and model
        processor = AutoProcessor.from_pretrained(str(final_model_path))
        model = WhisperForConditionalGeneration.from_pretrained(
            str(final_model_path), 
            torch_dtype=torch_dtype
        ).to(device)
        model.eval()
        
        # Load validation dataset offline
        from benchmark.dataset import load_asr_dataset, _load_audio_data
        dataset = load_asr_dataset(lang=lang, data_type=data_type)
        val_set = dataset["val"]
        
        references = []
        hypotheses = []
        
        # 4. Run batched inference loop
        batch_size = 16
        valid_entries = []
        for entry in val_set:
            ref_text = entry["text"].strip()
            if ref_text:
                valid_entries.append(entry)
                
        pbar = tqdm(total=len(valid_entries), desc="Transcribing (Batched)")
        for i in range(0, len(valid_entries), batch_size):
            batch_entries = valid_entries[i : i + batch_size]
            
            # Extract and stack audio features
            batch_features = []
            valid_batch_entries = []
            
            for entry in batch_entries:
                try:
                    array, sr = _load_audio_data(entry["audio"])
                    # WhisperProcessor feature extractor outputs shape (1, 80, 3000)
                    feat = processor(array, sampling_rate=sr, return_tensors="pt").input_features[0]
                    batch_features.append(feat)
                    valid_batch_entries.append(entry)
                except Exception as e:
                    print(f"[WARN] Failed to process audio for batch: {e}")
                    
            if not batch_features:
                pbar.update(len(batch_entries))
                continue
                
            try:
                # Stack features into shape (N, 80, 3000)
                input_features = torch.stack(batch_features).to(device)
                if device == "cuda":
                    input_features = input_features.to(torch.float16)
                    
                with torch.no_grad():
                    forced_decoder_ids = processor.get_decoder_prompt_ids(language=lang, task="transcribe")
                    generated_ids = model.generate(
                        input_features=input_features,
                        forced_decoder_ids=forced_decoder_ids
                    )
                    transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    
                for entry, transcription in zip(valid_batch_entries, transcriptions):
                    norm_ref = normalize_text(entry["text"], lang)
                    norm_hyp = normalize_text(transcription, lang)
                    
                    if norm_ref:
                        references.append(norm_ref)
                        hypotheses.append(norm_hyp)
            except Exception as e:
                print(f"[WARN] Failed batched generation: {e}")
                
            pbar.update(len(batch_entries))
        pbar.close()
                
        if not references:
            print("[ERROR] No references transcribed. Skipping.")
            continue
            
        # Compute normalized metrics
        wer = jiwer.wer(references, hypotheses)
        cer = jiwer.cer(references, hypotheses)
        
        print(f"  Normalized WER: {wer:.2%}")
        print(f"  Normalized CER: {cer:.2%}")
        
        # Estimate hours of training data used
        total_clips = len(dataset["train"])
        sub_clips = int(total_clips * fraction)
        hours = estimate_hours(sub_clips, lang)
        
        results.append({
            "Fraction": fraction,
            "Total_Clips": sub_clips,
            "Hours": hours,
            "WER": wer,
            "CER": cer
        })
        
    if not results:
        print("[ERROR] No results generated.")
        return
        
    # Save new statistics
    csv_path = output_dir / f"scaling_{lang}_{model_type}_{data_type}.csv"
    with open(csv_path, "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=["Fraction", "Total_Clips", "Hours", "WER", "CER"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nNormalized results saved to {csv_path}")
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))
    
    # Plot new curves
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(df["Hours"], df["WER"] * 100, marker='o', linewidth=2, label="WER (%)", color="crimson")
        plt.plot(df["Hours"], df["CER"] * 100, marker='s', linewidth=2, label="CER (%)", color="royalblue")
        
        plt.title(f"VALSEA ASR Scaling Laws (Normalized) - {lang.upper()} ({model_type.upper()})", fontsize=14, fontweight='bold')
        plt.xlabel("Dataset Size (Hours of Audio)", fontsize=12)
        plt.ylabel("Error Rate (%)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=11)
        
        plot_path = output_dir / f"scaling_{lang}_{model_type}_{data_type}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Normalized visualization curve saved to {plot_path}")
    except Exception as e:
        print(f"[WARN] Matplotlib plotting failed: {e}")

@click.command()
@click.option("--lang", default="zh", type=click.Choice(["zh", "hi"]))
@click.option("--model", "model_type", default="whisper")
@click.option("--data-type", default="synthetic")
@click.option("--device", default="cuda")
def main(**kwargs):
    re_evaluate_sweeps(**kwargs)

if __name__ == "__main__":
    main()
