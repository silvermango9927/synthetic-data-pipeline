"""Utility script to pre-download and cache Hugging Face models for offline ASR training."""
import click
from transformers import WhisperProcessor, WhisperForConditionalGeneration

@click.command()
@click.option("--model-name", default="openai/whisper-large-v3", help="Hugging Face model ID to cache")
def download_model(model_name: str):
    print(f"\n====================================================")
    # 1. Download processor
    print(f"Downloading and caching processor for: {model_name}...")
    processor = WhisperProcessor.from_pretrained(model_name)
    processor.save_pretrained(f"outputs/benchmark/cache/{model_name.replace('/', '_')}_processor")
    
    # 2. Download model weights
    print(f"Downloading and caching model weights for: {model_name}...")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.save_pretrained(f"outputs/benchmark/cache/{model_name.replace('/', '_')}_model")
    
    print(f"\n[SUCCESS] Model and processor for {model_name} have been fully cached locally!")
    print(f"  - Local model copy: outputs/benchmark/cache/{model_name.replace('/', '_')}_model")
    print(f"  - Local processor copy: outputs/benchmark/cache/{model_name.replace('/', '_')}_processor")
    print(f"  - Hugging Face cache directory: ~/.cache/huggingface/hub/")
    print(f"====================================================\n")

if __name__ == "__main__":
    download_model()
