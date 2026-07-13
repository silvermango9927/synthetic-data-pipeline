"""ASR Fine-Tuning entrypoint for Whisper, Qwen2-Audio, and Wav2Vec2/MMS models."""
import os
import sys
import click
import torch
from transformers import (
    AutoProcessor,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoModelForPreTraining,
    Qwen2AudioForConditionalGeneration,
    AutoModelForCTC,
    Wav2Vec2Processor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from benchmark.config import TrainConfig
from benchmark.dataset import load_asr_dataset, get_preprocess_fn, ASRDataCollator

def print_model_trainable_parameters(model):
    """Utility to print percentage of trainable parameters."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Trainable params: {trainable_params} | "
        f"All params: {all_param} | "
        f"Trainable %: {100 * trainable_params / all_param:.4f}%"
    )

def train_asr(cfg: TrainConfig):
    """Executes full training process for target model."""
    print(f"\n--- Initializing Training: {cfg.model_name} ({cfg.lang}) ---")
    print(f"  Data Mode: {cfg.data_type} (fraction={cfg.fraction})")
    
    # 1. Load processor based on model type
    print("Loading processor...")
    if cfg.model_type == "ctc":
        processor = Wav2Vec2Processor.from_pretrained(cfg.model_name)
    elif "qwen" in cfg.model_name.lower() or cfg.model_type == "qwen":
        processor = AutoProcessor.from_pretrained(cfg.model_name)
    else:
        processor = WhisperProcessor.from_pretrained(cfg.model_name, language=cfg.lang, task="transcribe")
        
    # 2. Load and preprocess dataset
    dataset = load_asr_dataset(
        lang=cfg.lang,
        data_type=cfg.data_type,
        fraction=cfg.fraction,
        dataset_name=cfg.dataset_name,
        val_fraction=cfg.val_fraction
    )
    
    preprocess_fn = get_preprocess_fn(processor, cfg.model_type)
    print("Preprocessing datasets (extracting speech features and tokenizing text)...")
    tokenized_dataset = dataset.map(
        preprocess_fn,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Tokenizing dataset"
    )
    
    # 3. Load pre-trained model
    print(f"Loading pre-trained model: {cfg.model_name}...")
    device_map = "auto" if cfg.device == "cuda" and torch.cuda.is_available() else None
    
    from transformers import BitsAndBytesConfig
    quant_kwargs = {}
    if cfg.load_in_8bit or cfg.load_in_4bit:
        if cfg.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        quant_kwargs["quantization_config"] = quantization_config
        
    if cfg.model_type == "ctc":
        # CTC models require target vocab size
        vocab_size = len(processor.tokenizer)
        model = AutoModelForCTC.from_pretrained(
            cfg.model_name,
            ctc_loss_reduction="mean",
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=vocab_size,
            device_map=device_map,
            **quant_kwargs
        )
    elif cfg.model_type == "qwen":
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            cfg.model_name,
            device_map=device_map,
            torch_dtype=torch.float16 if (cfg.load_in_8bit or cfg.load_in_4bit or cfg.fp16) else torch.float32,
            **quant_kwargs
        )
    else:  # whisper
        model = WhisperForConditionalGeneration.from_pretrained(
            cfg.model_name,
            device_map=device_map,
            torch_dtype=torch.float16 if (cfg.load_in_8bit or cfg.load_in_4bit or cfg.fp16) else torch.float32,
            **quant_kwargs
        )
        # Configure model generation configs
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        
        # Monkey-patch forward method to pop 'input_ids' and 'inputs_embeds' passed by PEFT
        # to prevent TypeError: WhisperDecoder() got multiple values for keyword arguments
        old_forward = model.forward
        def patched_forward(*args, **kwargs):
            kwargs.pop("input_ids", None)
            kwargs.pop("inputs_embeds", None)
            return old_forward(*args, **kwargs)
        model.forward = patched_forward
        
    # 4. Integrate PEFT / LoRA
    if cfg.use_lora:
        print("Integrating Parameter-Efficient Fine-Tuning (LoRA)...")
        # Handle kbit training setup for Qwen/large models if using quantization
        if cfg.device == "cuda" and (cfg.load_in_8bit or cfg.load_in_4bit):
            model = prepare_model_for_kbit_training(model)
            
        peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.lora_target_modules,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM" if cfg.model_type == "qwen" else "SEQ_2_SEQ_LM"
        )
        model = get_peft_model(model, peft_config)
        print_model_trainable_parameters(model)
        
    # 5. Define Collator & Training Arguments
    data_collator = ASRDataCollator(processor, cfg.model_type)
    
    print("Setting training arguments...")
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    if cfg.model_type == "ctc":
        training_args = TrainingArguments(
            output_dir=cfg.output_dir,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            eval_strategy="epoch" if "val" in tokenized_dataset else "no",
            save_strategy="epoch" if "val" in tokenized_dataset else "no",
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            num_train_epochs=cfg.epochs,
            warmup_ratio=cfg.warmup_ratio,
            logging_steps=cfg.logging_steps,
            fp16=cfg.fp16 and torch.cuda.is_available(),
            save_total_limit=cfg.save_total_limit,
            report_to="none",
            remove_unused_columns=False
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset.get("val"),
            data_collator=data_collator,
        )
    else:  # Whisper & Qwen (Seq2Seq)
        training_args = Seq2SeqTrainingArguments(
            output_dir=cfg.output_dir,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            num_train_epochs=cfg.epochs,
            eval_strategy="epoch" if "val" in tokenized_dataset else "no",
            save_strategy="epoch" if "val" in tokenized_dataset else "no",
            predict_with_generate=True,
            generation_max_length=225,
            logging_steps=cfg.logging_steps,
            fp16=cfg.fp16 and torch.cuda.is_available(),
            save_total_limit=cfg.save_total_limit,
            report_to="none",
            remove_unused_columns=False
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset.get("val"),
            data_collator=data_collator,
        )
        
    # 6. Execute training
    print("Starting model fine-tuning...")
    train_result = trainer.train()
    
    # Save final model checkpoint & processor
    print("Saving final model checkpoint...")
    trainer.save_model(os.path.join(cfg.output_dir, "final_model"))
    processor.save_pretrained(os.path.join(cfg.output_dir, "final_model"))
    
    print("\n--- Training Complete successfully! ---")
    return train_result.global_step, train_result.training_loss

@click.command()
@click.option("--model", "model_type", type=click.Choice(["whisper", "qwen", "ctc"]), default="whisper")
@click.option("--model-name", default="openai/whisper-tiny")
@click.option("--lang", default="zh", type=click.Choice(["zh", "hi"]))
@click.option("--data-type", default="synthetic", type=click.Choice(["synthetic", "augmented", "both"]))
@click.option("--fraction", default=1.0, type=float)
@click.option("--epochs", default=3, type=int)
@click.option("--batch-size", default=4, type=int)
@click.option("--lr", "learning_rate", default=5e-5, type=float)
@click.option("--use-lora/--no-lora", default=True)
@click.option("--load-in-8bit", is_flag=True, default=False)
@click.option("--load-in-4bit", is_flag=True, default=False)
@click.option("--device", default="cuda")
@click.option("--output-dir", default="outputs/benchmark/checkpoints")
@click.option("--dataset-name", default=None, help="Optionally override default HF dataset ID")
def main(**kwargs):
    cfg = TrainConfig(**kwargs)
    train_asr(cfg)

if __name__ == "__main__":
    main()
