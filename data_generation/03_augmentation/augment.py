"""Apply audio augmentations to make synthetic speech more realistic."""
import json
import shutil
from pathlib import Path

import click
import numpy as np
import soundfile as sf
from tqdm import tqdm

try:
    from audiomentations import (
        AddBackgroundNoise,
        ApplyImpulseResponse,
        BandPassFilter,
        Compose,
        Gain,
        Mp3Compression,
        PitchShift,
        TimeStretch,
    )
    _AUDIOMENTATIONS_AVAILABLE = True
except ImportError:
    _AUDIOMENTATIONS_AVAILABLE = False


def _identity(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """No-op augmenter used when audiomentations is not installed."""
    return samples


def build_augmentation_pipeline(noise_bank: str):
    """Build the augmentation pipeline.

    Falls back to a no-op if audiomentations is not installed, and skips
    noise/RIR transforms if the noise bank directories are empty.
    """
    if not _AUDIOMENTATIONS_AVAILABLE:
        print(
            "[WARN] audiomentations not installed — augmentation is a file copy only.\n"
            "  To enable: sudo xcodebuild -license && pip install -e '.[augment]'"
        )
        return _identity

    transforms = []
    noise_path = Path(noise_bank)

    # Background noise
    noise_dir = noise_path / "ambient"
    if noise_dir.exists() and list(noise_dir.glob("*.wav")):
        transforms.append(
            AddBackgroundNoise(sounds_path=str(noise_dir), min_snr_db=10, max_snr_db=25, p=0.7)
        )
    else:
        print(f"[WARN] No ambient noise files in {noise_dir}. Skipping noise augmentation.")
        print("  Download MUSAN: wget https://openslr.org/resources/17/musan.tar.gz")

    # Room impulse response (reverb)
    rir_dir = noise_path / "rir"
    if rir_dir.exists() and list(rir_dir.glob("*.wav")):
        transforms.append(ApplyImpulseResponse(ir_path=str(rir_dir), p=0.5))
    else:
        print(f"[WARN] No RIR files in {rir_dir}. Skipping reverb augmentation.")
        print("  Download: wget https://openslr.org/resources/28/rirs_noises.zip")

    # These always work (no external files needed)
    transforms.extend([
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.4),
        PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
        Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
        Mp3Compression(min_bitrate=32, max_bitrate=64, p=0.3),
        BandPassFilter(min_center_freq=200, max_center_freq=4000, p=0.2),
    ])

    return Compose(transforms)


@click.command()
@click.option("--input-dir", required=True, help="Directory with clean WAVs")
@click.option("--output-dir", required=True, help="Output directory for augmented WAVs")
@click.option("--noise-bank", default="data_generation/03_augmentation/noise_bank", help="Noise bank directory")
@click.option("--variants", default=2, help="Number of augmented variants per clean file")
@click.option("--sample-rate", default=16000, help="Target sample rate")
def main(input_dir: str, output_dir: str, noise_bank: str, variants: int, sample_rate: int):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    wav_files = list(in_path.glob("*.wav"))
    if not wav_files:
        print(f"[ERROR] No WAV files found in {in_path}")
        return

    augmenter = build_augmentation_pipeline(noise_bank)

    # Load clean manifest
    clean_manifest_path = in_path / "manifest_clean.jsonl"
    clean_manifest: dict = {}
    if clean_manifest_path.exists():
        with open(clean_manifest_path) as f:
            for line in f:
                entry = json.loads(line)
                clean_manifest[Path(entry["audio_filepath"]).name] = entry

    augmented_manifest = []

    print(f"Augmenting {len(wav_files)} files × {variants} variants...")

    for wav_path in tqdm(wav_files):
        audio, sr = sf.read(str(wav_path), dtype="float32")

        if sr != sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            sr = sample_rate

        for v in range(variants):
            aug_audio = augmenter(samples=audio, sample_rate=sr)

            aug_filename = f"{wav_path.stem}_aug{v:02d}.wav"
            aug_path = out_path / aug_filename
            sf.write(str(aug_path), aug_audio, sr)

            base_entry = clean_manifest.get(wav_path.name, {})
            aug_label = "passthrough" if not _AUDIOMENTATIONS_AVAILABLE else f"variant_{v}"
            augmented_manifest.append({
                "audio_filepath": str(aug_path),
                "text": base_entry.get("text", ""),
                "duration": len(aug_audio) / sr,
                "language": base_entry.get("language", ""),
                "source": "synthetic",
                "voice_id": base_entry.get("voice_id", ""),
                "augmentation": aug_label,
            })

    manifest_path = out_path / "manifest_augmented.jsonl"
    with open(manifest_path, "w") as f:
        for entry in augmented_manifest:
            f.write(json.dumps(entry) + "\n")

    print(f"Created {len(augmented_manifest)} augmented files. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
