"""Fallback augmenter using librosa + scipy when audiomentations won't build.

Mirrors the non-noise/non-RIR transforms in 03_augmentation/augment.py:
time-stretch, pitch-shift, gain, bandpass, light additive noise.
Use this only when audiomentations is unavailable; the canonical pipeline is
03_augmentation/augment.py.
"""
import json
import random
from pathlib import Path

import click
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt
from tqdm import tqdm


def _bandpass(audio: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
    nyq = sr / 2
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, audio).astype(np.float32)


def augment_one(audio: np.ndarray, sr: int, rng: random.Random) -> tuple[np.ndarray, list[str]]:
    """Apply a randomized chain of transforms. Returns (audio, applied_labels)."""
    applied: list[str] = []
    out = audio.copy()

    if rng.random() < 0.6:
        rate = rng.uniform(0.9, 1.1)
        out = librosa.effects.time_stretch(out, rate=rate)
        applied.append(f"time_stretch={rate:.3f}")

    if rng.random() < 0.5:
        steps = rng.uniform(-2, 2)
        out = librosa.effects.pitch_shift(out, sr=sr, n_steps=steps)
        applied.append(f"pitch_shift={steps:+.2f}st")

    if rng.random() < 0.4:
        low = rng.uniform(150, 350)
        high = rng.uniform(3500, 4500)
        out = _bandpass(out, sr, low, high)
        applied.append(f"bandpass={int(low)}-{int(high)}Hz")

    if rng.random() < 0.6:
        snr_db = rng.uniform(15, 30)
        sig_power = float(np.mean(out**2)) + 1e-12
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), size=out.shape).astype(np.float32)
        out = out + noise
        applied.append(f"noise_snr={snr_db:.1f}dB")

    gain_db = rng.uniform(-6, 6)
    out = out * (10 ** (gain_db / 20))
    applied.append(f"gain={gain_db:+.2f}dB")

    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.99:
        out = out * (0.99 / peak)
        applied.append("peak_normalize")

    return out.astype(np.float32), applied


@click.command()
@click.option("--input-dir", required=True)
@click.option("--output-dir", required=True)
@click.option("--variants", default=2)
@click.option("--limit", default=0, help="Max clean files to process; 0 = all")
@click.option("--sample-rate", default=16000)
@click.option("--seed", default=42)
def main(input_dir: str, output_dir: str, variants: int, limit: int, sample_rate: int, seed: int):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(in_path.glob("*.wav"))
    if limit > 0:
        wav_files = wav_files[:limit]
    if not wav_files:
        print(f"[ERROR] No WAVs in {in_path}")
        return

    clean_manifest_path = in_path / "manifest_clean.jsonl"
    clean_manifest: dict = {}
    if clean_manifest_path.exists():
        with open(clean_manifest_path) as f:
            for line in f:
                e = json.loads(line)
                clean_manifest[Path(e["audio_filepath"]).name] = e

    rng = random.Random(seed)
    np.random.seed(seed)

    augmented = []
    print(f"Augmenting {len(wav_files)} files × {variants} variants → {out_path}")

    for wav_path in tqdm(wav_files):
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            sr = sample_rate

        for v in range(variants):
            aug_audio, applied = augment_one(audio, sr, rng)
            aug_filename = f"{wav_path.stem}_aug{v:02d}.wav"
            aug_path = out_path / aug_filename
            sf.write(str(aug_path), aug_audio, sr)

            base = clean_manifest.get(wav_path.name, {})
            augmented.append({
                "audio_filepath": str(aug_path),
                "text": base.get("text", ""),
                "duration": len(aug_audio) / sr,
                "language": base.get("language", ""),
                "source": "synthetic",
                "voice_id": base.get("voice_id", ""),
                "augmentation": "+".join(applied),
            })

    manifest_path = out_path / "manifest_augmented.jsonl"
    with open(manifest_path, "w") as f:
        for entry in augmented:
            f.write(json.dumps(entry) + "\n")
    print(f"Wrote {len(augmented)} augmented files. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
