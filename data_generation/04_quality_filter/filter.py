"""Three-layer quality filter: UTMOS + ASR roundtrip + acoustic sanity."""
import json
from pathlib import Path

import click
import numpy as np
import soundfile as sf
from tqdm import tqdm


def check_duration_sanity(
    audio_path: str, text: str, min_chars_per_sec: float = 4.0, max_chars_per_sec: float = 20.0
) -> tuple[bool, str]:
    """Check if audio duration is reasonable for the text length."""
    info = sf.info(audio_path)
    duration = info.duration
    char_count = len(text)

    if duration < 0.5:
        return False, "too_short"
    if duration > 60:
        return False, "too_long"

    chars_per_sec = char_count / duration if duration > 0 else 0
    if chars_per_sec < min_chars_per_sec or chars_per_sec > max_chars_per_sec:
        return False, f"rate_mismatch_{chars_per_sec:.1f}cps"

    # Check for excessive leading silence
    audio, sr = sf.read(audio_path, dtype="float32")
    energy = np.abs(audio)

    threshold = np.max(energy) * 0.02
    leading_silence = np.argmax(energy > threshold) / sr
    if leading_silence > 2.0:
        return False, f"leading_silence_{leading_silence:.1f}s"

    return True, "ok"


def score_utmos(audio_path: str) -> float:
    """Score audio naturalness using UTMOS. Returns 1.0-5.0."""
    try:
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForSequenceClassification

        # Lazy load — this is slow on first call
        if not hasattr(score_utmos, "_model"):
            score_utmos._model = AutoModelForSequenceClassification.from_pretrained(
                "sarulab-speech/utmos22_strong", trust_remote_code=True
            )
            score_utmos._fe = AutoFeatureExtractor.from_pretrained(
                "sarulab-speech/utmos22_strong", trust_remote_code=True
            )

        audio, sr = sf.read(audio_path, dtype="float32")
        if sr != 16000:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        inputs = score_utmos._fe(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            score = score_utmos._model(**inputs).logits.squeeze().item()
        return float(np.clip(score, 1.0, 5.0))
    except Exception as e:
        print(f"[WARN] UTMOS scoring failed: {e}. Returning 3.0 as default.")
        return 3.0


def roundtrip_wer(audio_path: str, expected_text: str, language: str) -> float:
    """Transcribe audio with Whisper and compute WER against expected text."""
    try:
        import jiwer
        import whisper

        if not hasattr(roundtrip_wer, "_model"):
            print("Loading Whisper large-v3-turbo... (first call only)")
            roundtrip_wer._model = whisper.load_model("large-v3-turbo")

        result = roundtrip_wer._model.transcribe(audio_path, language=language, fp16=True)
        hypothesis = result["text"].strip()

        wer = jiwer.wer(expected_text.strip(), hypothesis)
        return float(wer)
    except Exception as e:
        print(f"[WARN] ASR roundtrip failed: {e}. Returning 1.0 (worst).")
        return 1.0


def roundtrip_wer_qwen2_audio(audio_path: str, expected_text: str, language: str) -> float:
    """Transcribe audio with Qwen2-Audio-7B-Instruct and compute WER.

    Used for Chinese QC roundtrip when the downstream ASR target is Qwen2-Audio.
    Requires ~14GB VRAM. Lazy-loads on first call.
    """
    try:
        import jiwer
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        if not hasattr(roundtrip_wer_qwen2_audio, "_model"):
            print("Loading Qwen/Qwen2-Audio-7B-Instruct... (first call only, ~14GB VRAM)")
            model_id = "Qwen/Qwen2-Audio-7B-Instruct"
            roundtrip_wer_qwen2_audio._processor = AutoProcessor.from_pretrained(model_id)
            roundtrip_wer_qwen2_audio._model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_id, device_map="auto", torch_dtype=torch.float16
            )

        processor = roundtrip_wer_qwen2_audio._processor
        model = roundtrip_wer_qwen2_audio._model

        audio, sr = sf.read(audio_path, dtype="float32")
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Transcribe the speech."
        inputs = processor(
            text=prompt, audios=audio, sampling_rate=16000, return_tensors="pt", padding=True
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids = generated_ids[:, inputs.input_ids.size(1):]
        hypothesis = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        wer = jiwer.wer(expected_text.strip(), hypothesis)
        return float(wer)
    except Exception as e:
        print(f"[WARN] Qwen2-Audio roundtrip failed: {e}. Returning 1.0 (worst).")
        return 1.0


WER_THRESHOLDS = {"vi": 0.15, "en": 0.25, "zh": 0.20, "hi": 0.25}


@click.command()
@click.option("--input-dir", required=True, help="Directory containing clean/ and augmented/ subdirs")
@click.option("--output", required=True, help="Output filtered manifest JSONL")
@click.option("--lang", required=True, help="Language code: en, vi, zh, or hi")
@click.option("--utmos-threshold", default=3.5, help="Minimum UTMOS score")
@click.option(
    "--wer-threshold",
    default=None,
    type=float,
    help="Max WER (auto per-language: vi 0.15, en 0.25, zh 0.20, hi 0.25)",
)
@click.option("--skip-utmos", is_flag=True, help="Skip UTMOS scoring (faster, less filtering)")
@click.option(
    "--skip-whisper",
    is_flag=True,
    help="Skip ASR roundtrip (faster, less filtering). Applies to whichever --qc-asr is selected.",
)
@click.option(
    "--qc-asr",
    type=click.Choice(["whisper", "qwen2-audio"]),
    default="whisper",
    help="ASR model used for the roundtrip WER check. qwen2-audio aligns with Chinese training target.",
)
def main(
    input_dir: str,
    output: str,
    lang: str,
    utmos_threshold: float,
    wer_threshold: float | None,
    skip_utmos: bool,
    skip_whisper: bool,
    qc_asr: str,
):
    if wer_threshold is None:
        wer_threshold = WER_THRESHOLDS.get(lang, 0.25)

    in_path = Path(input_dir)

    # Collect all manifests from clean/ and augmented/ subdirs
    all_entries = []
    for manifest_file in in_path.rglob("manifest_*.jsonl"):
        with open(manifest_file) as f:
            for line in f:
                all_entries.append(json.loads(line))

    if not all_entries:
        print(f"[ERROR] No manifest files found in {in_path}")
        return

    print(f"Filtering {len(all_entries)} samples (UTMOS≥{utmos_threshold}, WER≤{wer_threshold})")

    passed = []
    rejected = {"utmos": 0, "wer": 0, "duration": 0, "missing": 0}

    for entry in tqdm(all_entries):
        audio_path = entry["audio_filepath"]
        text = entry.get("text", "")

        if not Path(audio_path).exists():
            rejected["missing"] += 1
            continue

        # Layer 1: Duration sanity (fast, do first)
        ok, reason = check_duration_sanity(audio_path, text)
        if not ok:
            rejected["duration"] += 1
            continue

        # Layer 2: UTMOS
        if not skip_utmos:
            mos = score_utmos(audio_path)
            if mos < utmos_threshold:
                rejected["utmos"] += 1
                continue
            entry["utmos_score"] = mos

        # Layer 3: ASR roundtrip
        if not skip_whisper:
            if qc_asr == "qwen2-audio":
                wer = roundtrip_wer_qwen2_audio(audio_path, text, lang)
            else:
                wer = roundtrip_wer(audio_path, text, lang)
            if wer > wer_threshold:
                rejected["wer"] += 1
                continue
            entry["roundtrip_wer"] = wer
            entry["qc_asr"] = qc_asr

        passed.append(entry)

    # Write filtered manifest
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for entry in passed:
            f.write(json.dumps(entry) + "\n")

    total = len(all_entries)
    print("\n=== Filter Results ===")
    print(f"  Total:    {total}")
    print(f"  Passed:   {len(passed)} ({len(passed)/total:.1%})")
    print(f"  Rejected: {sum(rejected.values())}")
    for reason, count in rejected.items():
        if count > 0:
            print(f"    - {reason}: {count}")
    print(f"  Output:   {out_path}")


if __name__ == "__main__":
    main()
