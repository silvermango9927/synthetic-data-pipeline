"""Configuration schema for ASR benchmarking and scaling laws."""
from pydantic import BaseModel
from typing import Optional, List

class TrainConfig(BaseModel):
    """Configuration for training ASR models."""
    model_name: str = "openai/whisper-tiny"  # whisper-tiny / Qwen2-Audio / MMS
    model_type: str = "whisper"              # whisper | qwen | ctc
    lang: str = "zh"                         # zh | hi
    
    # Dataset
    dataset_name: Optional[str] = None       # HF dataset path or local manifest JSONL
    data_type: str = "synthetic"             # synthetic (clean) | augmented | both
    fraction: float = 1.0                    # Fraction of data to use (0.0 to 1.0)
    train_split: str = "train"
    val_split: str = "val"
    val_fraction: float = 0.05               # Heldout split ratio if val not in dataset
    
    # Training Parameters
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 2
    fp16: bool = True
    
    # PEFT/LoRA options (required for large models like Qwen2-Audio-7B)
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = ["q_proj", "v_proj"]
    
    # Hardware/Saving
    device: str = "cuda"                     # cuda | cpu
    output_dir: str = "outputs/benchmark/checkpoints"
    logging_steps: int = 10
    eval_steps: int = 50
    save_total_limit: int = 1
