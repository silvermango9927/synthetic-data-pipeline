"""ASR Evaluation script to compute WER and CER on validation/test datasets."""
import os
import click
import torch
import jiwer
from tqdm import tqdm
from transformers import AutoProcessor, WhisperForConditionalGeneration, AutoModelForCTC, Qwen2AudioForConditionalGeneration, Wav2Vec2Processor
from benchmark.dataset import load_asr_dataset, _load_audio_data

def evaluate_asr(
    model_path: str,
    model_type: str = "whisper",
    lang: str = "zh",
    dataset_name: str = None,
    data_type: str = "synthetic",
    device: str = "cuda"
):
    """Evaluates fine-tuned model against validation set and computes WER & CER."""
    print(f"\n--- Initializing Evaluation: {model_path} ---")
    
    # 1. Resolve active device
    device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 2. Load fine-tuned processor and model
    print("Loading model and processor...")
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        print("PEFT adapter found. Loading base model with adapters...")
        from peft import PeftConfig, PeftModel
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model_name = peft_config.base_model_name_or_path
        
        processor = AutoProcessor.from_pretrained(model_path)
        if model_type == "qwen":
            base_model = Qwen2AudioForConditionalGeneration.from_pretrained(base_model_name, torch_dtype=torch_dtype)
        else:
            base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name, torch_dtype=torch_dtype)
        model = PeftModel.from_pretrained(base_model, model_path).to(device)
    else:
        if model_type == "ctc":
            processor = Wav2Vec2Processor.from_pretrained(model_path)
            model = AutoModelForCTC.from_pretrained(model_path).to(device)
        elif model_type == "qwen":
            processor = AutoProcessor.from_pretrained(model_path)
            model = Qwen2AudioForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch_dtype).to(device)
        else:  # whisper
            processor = AutoProcessor.from_pretrained(model_path)
            model = WhisperForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch_dtype).to(device)
        
    model.eval()
    
    # 3. Load dataset validation split
    dataset = load_asr_dataset(
        lang=lang,
        data_type=data_type,
        dataset_name=dataset_name
    )
    val_set = dataset["val"]
    print(f"Validation dataset size: {len(val_set)} samples")
    
    references = []
    hypotheses = []
    
    # 4. Run inference loop
    for entry in tqdm(val_set, desc="Transcribing"):
        audio = entry["audio"]
        ref_text = entry["text"].strip()
        
        # Guard empty ground truth
        if not ref_text:
            continue
            
        try:
            array, sr = _load_audio_data(audio)
            with torch.no_grad():
                if model_type == "ctc":
                    inputs = processor(array, sampling_rate=sr, return_tensors="pt").to(device)
                    logits = model(inputs.input_values).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = processor.batch_decode(predicted_ids)[0]
                elif model_type == "qwen":
                    inputs = processor(audio=array, sampling_rate=sr, return_tensors="pt").to(device)
                    if device == "cuda":
                        inputs = {k: v.to(torch.float16) if v.dtype == torch.float32 else v for k, v in inputs.items()}
                    generated_ids = model.generate(**inputs, max_new_tokens=225)
                    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                else:  # whisper
                    inputs = processor(array, sampling_rate=sr, return_tensors="pt").to(device)
                    if device == "cuda":
                        inputs = {k: v.to(torch.float16) if v.dtype == torch.float32 else v for k, v in inputs.items()}
                    forced_decoder_ids = processor.get_decoder_prompt_ids(language=lang, task="transcribe")
                    generated_ids = model.generate(**inputs, forced_decoder_ids=forced_decoder_ids)
                    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
            references.append(ref_text)
            hypotheses.append(transcription.strip())
        except Exception as e:
            print(f"[WARN] Failed to transcribe sample: {e}")
            
    if not references:
        print("[ERROR] No samples transcribed.")
        return 1.0, 1.0
        
    # 5. Compute error rates (WER & CER)
    print("Computing metrics...")
    
    # Clean references/hypotheses for cleaner WER computation
    wer = jiwer.wer(references, hypotheses)
    
    # Compute character-level error rate (CER)
    # CER is simply character-wise WER
    cer = jiwer.cer(references, hypotheses)
    
    print("\n================ EVALUATION METRICS ================")
    print(f"  Word Error Rate (WER)      : {wer:.4f} ({wer:.2%})")
    print(f"  Character Error Rate (CER) : {cer:.4f} ({cer:.2%})")
    print("====================================================")
    
    # Print a few examples
    print("\nSample Transcriptions (Reference vs Hypothesis):")
    for i in range(min(3, len(references))):
        try:
            print(f"  {i+1}. [REF]: {references[i]}")
            print(f"     [HYP]: {hypotheses[i]}")
        except UnicodeEncodeError:
            # Fallback for Windows CP1252 terminal
            ref_safe = references[i].encode('ascii', errors='replace').decode('ascii')
            hyp_safe = hypotheses[i].encode('ascii', errors='replace').decode('ascii')
            print(f"  {i+1}. [REF] (safe-print): {ref_safe}")
            print(f"     [HYP] (safe-print): {hyp_safe}")
        print()
        
    return float(wer), float(cer)

@click.command()
@click.option("--model-path", required=True, help="Path to fine-tuned model checkpoint directory")
@click.option("--model", "model_type", type=click.Choice(["whisper", "qwen", "ctc"]), default="whisper")
@click.option("--lang", default="zh", type=click.Choice(["zh", "hi"]))
@click.option("--data-type", default="synthetic", type=click.Choice(["synthetic", "augmented", "both"]))
@click.option("--dataset-name", default=None)
@click.option("--device", default="cuda")
def main(**kwargs):
    evaluate_asr(**kwargs)

if __name__ == "__main__":
    main()
