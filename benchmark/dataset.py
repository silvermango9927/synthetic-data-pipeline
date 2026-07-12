"""Dataset utilities for loading and preprocessing HF datasets and NeMo ASR manifests."""
import os
from datasets import Dataset, load_dataset, Audio, DatasetDict
import json
import random
import torch
import soundfile as sf
import numpy as np

def load_asr_dataset(
    lang: str,
    data_type: str = "synthetic",  # synthetic (clean) | augmented | both
    fraction: float = 1.0,
    dataset_name: str = None,      # e.g., "silvermango9927/synthetic-asr-zh"
    val_fraction: float = 0.05,
    seed: int = 42
):
    """Loads ASR datasets from HF hub or local manifests, handles train/val split and data fractionation."""
    from dotenv import load_dotenv
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    use_local = (dataset_name == "local")
    dataset = None

    if not use_local:
        if not dataset_name:
            dataset_name = f"silvermango9927/synthetic-asr-{lang}"
        
        # Check for local cloned repository
        from pathlib import Path
        local_repo_dir = Path("outputs/hf_datasets") / f"synthetic-asr-{lang}"
        
        # Resolve config names
        configs = []
        if data_type in ("synthetic", "both"):
            configs.append("long_clean")
        if data_type in ("augmented", "both"):
            configs.append("long_augmented")
            
        ds_list = []
        
        if local_repo_dir.exists():
            print(f"Found local cloned repository at {local_repo_dir}. Loading offline...")
            for cfg in configs:
                train_file = local_repo_dir / "data" / cfg / "manifest.jsonl"
                val_file = local_repo_dir / "data" / cfg / "val.jsonl"
                
                if train_file.exists() and val_file.exists():
                    try:
                        ds = load_dataset(
                            "json",
                            data_files={
                                "train": str(train_file),
                                "val": str(val_file)
                            },
                            encoding="utf-8"
                        )
                        # Resolve relative audio paths to absolute paths
                        audio_base = local_repo_dir / "data" / cfg
                        def resolve_paths(batch):
                            batch["audio"] = str((audio_base / batch["audio_filepath"]).resolve())
                            batch["audio_filepath"] = batch["audio"]
                            return batch
                        
                        ds = ds.map(resolve_paths, num_proc=1, desc=f"Resolving local paths for {cfg}")
                        ds_list.append(ds)
                    except Exception as e:
                        print(f"[WARN] Failed to load local config {cfg}: {e}")
        else:
            print(f"Loading dataset: {dataset_name} for type: {data_type} from HF Hub...")
            for cfg in configs:
                try:
                    ds = load_dataset(dataset_name, cfg, token=token)
                    ds_list.append(ds)
                except Exception as e:
                    print(f"[WARN] Failed to load config {cfg} from HF {dataset_name}: {e}")
                    
        if ds_list:
            # Combine datasets if 'both' is selected
            if len(ds_list) == 2:
                from datasets import concatenate_datasets
                train_combined = concatenate_datasets([ds_list[0]["train"], ds_list[1]["train"]])
                val_combined = concatenate_datasets([ds_list[0]["val"], ds_list[1]["val"]])
                dataset = DatasetDict({"train": train_combined, "val": val_combined})
            else:
                dataset = ds_list[0]
            
            # Only cast to Audio feature if we loaded online from HF Hub (local JSON loader returns paths as raw strings)
            if not local_repo_dir.exists():
                dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    if dataset is None:
        print("[INFO] HF loading failed or dataset_name='local'. Falling back to local directory scanning...")
        # Fallback to local files in outputs/
        from pathlib import Path
        lang_map = {"zh": "chinese", "hi": "hindi"}
        lang_dir = Path("outputs") / lang_map.get(lang, "chinese") / "long"
        
        audio_paths = []
        if data_type in ("synthetic", "both"):
            clean_dir = lang_dir / "clean"
            if clean_dir.exists():
                audio_paths.extend(list(clean_dir.glob("*.wav")))
        if data_type in ("augmented", "both"):
            aug_dir = lang_dir / "augmented"
            if aug_dir.exists():
                audio_paths.extend(list(aug_dir.glob("*.wav")))
                
        if not audio_paths:
            raise ValueError(f"No local WAV files found in {lang_dir}")
            
        print(f"Found {len(audio_paths)} local WAV files. Building local dataset...")
        
        # Build dataset entries
        records = []
        dummy_text = "这是一个本地测试句子" if lang == "zh" else "यह एक स्थानीय परीक्षण वाक्य है"
        
        for p in audio_paths:
            try:
                info = sf.info(str(p))
                records.append({
                    "audio": str(p.resolve()),
                    "audio_filepath": str(p.resolve()),
                    "text": dummy_text,
                    "duration": info.duration,
                    "language": lang,
                    "source": "synthetic",
                    "voice_id": "local",
                    "augmentation": "none"
                })
            except Exception:
                continue
                
        # Split train/val deterministically
        random.seed(seed)
        random.shuffle(records)
        val_count = max(1, int(len(records) * val_fraction))
        val_records = records[:val_count]
        train_records = records[val_count:]
        
        train_ds = Dataset.from_dict({k: [r[k] for r in train_records] for k in records[0].keys()})
        val_ds = Dataset.from_dict({k: [r[k] for r in val_records] for k in records[0].keys()})
        dataset = DatasetDict({"train": train_ds, "val": val_ds})
    
    # Deterministic fractionation for scaling laws
    if fraction < 1.0:
        for split in ("train",):
            ds_split = dataset[split]
            total_len = len(ds_split)
            subset_len = int(total_len * fraction)
            print(f"Fractionating {split} split from {total_len} down to {subset_len} samples ({fraction:.1%})...")
            # Select deterministic indices using seed
            random.seed(seed)
            indices = list(range(total_len))
            random.shuffle(indices)
            selected_indices = indices[:subset_len]
            dataset[split] = ds_split.select(selected_indices)
            
    return dataset

class ASRDataCollator:
    """Collator for dynamic padding of speech inputs and text labels."""
    def __init__(self, processor, model_type: str = "whisper"):
        self.processor = processor
        self.model_type = model_type
        
    def __call__(self, features):
        if features and len(features) > 0:
            print(f"[DEBUG] ASRDataCollator keys: {list(features[0].keys())}")
        # Extract features (spectrograms / raw waveforms)
        if self.model_type == "ctc":
            input_features = [{"input_values": feature["input_values"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        elif self.model_type == "qwen":
            # Qwen2-Audio / Multimodal sequence preparation
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        else:  # whisper
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            
        # Process and pad target labels
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        # Replace padding token id with -100 to ignore loss calculation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        
        return batch

def _load_audio_data(audio):
    """Dynamically loads audio from HF dictionary or local filepath."""
    if isinstance(audio, dict):
        return audio["array"], audio["sampling_rate"]
    else:
        # Load local filepath string
        import soundfile as sf
        import librosa
        array, sr = sf.read(str(audio))
        if sr != 16000:
            array = librosa.resample(array, orig_sr=sr, target_sr=16000)
            sr = 16000
        return array, sr

def get_preprocess_fn(processor, model_type: str = "whisper"):
    """Returns preprocessing function customized for model type."""
    def preprocess_whisper(batch):
        array, sr = _load_audio_data(batch["audio"])
        batch["input_features"] = processor(
            array, sampling_rate=sr
        ).input_features[0]
        labels = processor.tokenizer(batch["text"]).input_ids
        if len(labels) > 448:
            labels = labels[:448]
        batch["labels"] = labels
        return batch
        
    def preprocess_qwen(batch):
        array, sr = _load_audio_data(batch["audio"])
        # Format for Qwen2AudioProcessor: accepts audio inputs directly
        inputs = processor(
            audio=array, sampling_rate=sr, text=batch["text"], return_tensors="pt"
        )
        batch["input_features"] = inputs.input_features[0]
        batch["labels"] = inputs.input_ids[0]
        return batch
        
    def preprocess_ctc(batch):
        array, sr = _load_audio_data(batch["audio"])
        batch["input_values"] = processor(
            array, sampling_rate=sr
        ).input_values[0]
        with processor.as_target_processor():
            batch["labels"] = processor(batch["text"]).input_ids
        return batch
        
    if model_type == "qwen":
        return preprocess_qwen
    elif model_type == "ctc":
        return preprocess_ctc
    return preprocess_whisper
