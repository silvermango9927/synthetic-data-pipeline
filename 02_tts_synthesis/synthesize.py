"""Synthesize audio from text corpus using TTS."""
import asyncio
import hashlib
import json
import random
from pathlib import Path

import click
import soundfile as sf
from tqdm import tqdm

# Default edge-tts voices per language (Microsoft Neural TTS, free, no server needed)
EDGE_VOICES = {
    "en": ["en-SG-LunaNeural", "en-SG-WayneNeural", "en-US-JennyNeural", "en-US-GuyNeural"],
    "vi": ["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"],
}


def get_voice_files(voice_bank_dir: str) -> list[Path]:
    """Get all audio files from voice bank directory."""
    vb = Path(voice_bank_dir)
    voices = list(vb.glob("*.wav")) + list(vb.glob("*.mp3")) + list(vb.glob("*.flac"))
    if not voices:
        raise FileNotFoundError(
            f"No audio files in {vb}. Add reference speaker WAVs (10-30s each).\n"
            f"  Tip: use --backend edge for a zero-config prototype (no voice bank needed)."
        )
    return voices


def synthesize_fish_speech(
    text: str,
    voice_path: Path,
    output_path: Path,
    api_url: str = "http://localhost:8080/v1/tts",
    language: str = "en",
) -> bool:
    """Synthesize using Fish Speech S2 API (local or cloud)."""
    import httpx

    try:
        with open(voice_path, "rb") as f:
            resp = httpx.post(
                api_url,
                files={"reference_audio": (voice_path.name, f, "audio/wav")},
                data={"text": text, "language": language},
                timeout=120.0,
            )
        if resp.status_code == 200:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(resp.content)
            return True
        else:
            print(f"[WARN] Fish Speech returned {resp.status_code}: {resp.text[:200]}")
            return False
    except httpx.ConnectError:
        print("[ERROR] Cannot connect to Fish Speech. Is the server running?")
        print("  Start with: docker run -p 8080:8080 --gpus all fishaudio/fish-speech:latest")
        return False


def synthesize_xtts(
    text: str,
    voice_path: Path,
    output_path: Path,
    language: str = "en",
) -> bool:
    """Synthesize using XTTS-v2 (local inference, fallback)."""
    try:
        from TTS.api import TTS
    except ImportError:
        raise ImportError("pip install -e '.[tts-local]'  # for XTTS-v2")

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tts.tts_to_file(
        text=text,
        speaker_wav=str(voice_path),
        language=language,
        file_path=str(output_path),
    )
    return True


def synthesize_edge_tts(
    text: str,
    voice: str,
    output_path: Path,
) -> bool:
    """Synthesize using Microsoft Edge TTS (free, no server or voice bank needed).

    Install: pip install edge-tts
    Voices: https://speech.microsoft.com/portal/voicegallery
    """
    try:
        import edge_tts
    except ImportError:
        raise ImportError("pip install edge-tts  # or: pip install -e '.[prototype]'")

    async def _run() -> None:
        communicate = edge_tts.Communicate(text, voice)
        # edge-tts outputs MP3; soundfile needs WAV — save to tmp then convert
        tmp_path = output_path.with_suffix(".mp3")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        await communicate.save(str(tmp_path))

        # Convert MP3 → WAV at 16kHz mono
        import librosa
        import numpy as np

        audio, sr = librosa.load(str(tmp_path), sr=16000, mono=True)
        sf.write(str(output_path), audio, sr)
        tmp_path.unlink(missing_ok=True)

    try:
        asyncio.run(_run())
        return True
    except Exception as e:
        print(f"[ERROR] edge-tts failed for voice {voice!r}: {e}")
        return False


def make_output_filename(text: str, voice_id: str, idx: int) -> str:
    text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"synth_{idx:06d}_{voice_id}_{text_hash}.wav"


@click.command()
@click.option("--corpus", required=True, help="Input JSONL corpus path")
@click.option(
    "--voice-bank",
    default="",
    help="Directory of reference speaker WAVs (not needed for --backend edge)",
)
@click.option("--output-dir", required=True, help="Output directory for WAVs")
@click.option("--lang", default="en", help="Language code: en (Singlish) or vi (Vietnamese)")
@click.option("--voices-per-sentence", default=3, help="Number of voice variants per sentence")
@click.option(
    "--backend",
    default="fish",
    type=click.Choice(["fish", "xtts", "edge"]),
    help="TTS backend. 'edge' requires no server or voice bank.",
)
@click.option("--api-url", default="http://localhost:8080/v1/tts", help="Fish Speech API URL")
@click.option("--max-sentences", default=None, type=int, help="Limit sentences to process")
def main(
    corpus: str,
    voice_bank: str,
    output_dir: str,
    lang: str,
    voices_per_sentence: int,
    backend: str,
    api_url: str,
    max_sentences: int | None,
):
    # --- resolve voice pool ---
    if backend == "edge":
        lang_key = lang.split("-")[0]  # "en-SG" → "en"
        edge_voices = EDGE_VOICES.get(lang_key, EDGE_VOICES["en"])
        voices: list = edge_voices
        print(f"edge-tts backend: using {len(voices)} voices for lang={lang!r}")
    else:
        if not voice_bank:
            raise click.UsageError("--voice-bank is required for fish and xtts backends.")
        voices = get_voice_files(voice_bank)
        print(f"Found {len(voices)} voice profiles in {voice_bank}")

    # Load corpus
    sentences = []
    with open(corpus) as f:
        for line in f:
            sentences.append(json.loads(line))

    if max_sentences:
        sentences = sentences[:max_sentences]

    n_voices = min(voices_per_sentence, len(voices))
    print(
        f"Synthesizing {len(sentences)} sentences × {n_voices} voices = "
        f"{len(sentences) * n_voices} audio files"
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = []

    for idx, item in enumerate(tqdm(sentences)):
        text = item["text"]
        selected = random.sample(voices, k=n_voices)

        for voice in selected:
            if backend == "edge":
                voice_id = voice.replace("-", "_")
                filename = make_output_filename(text, voice_id, idx)
                output_path = out / filename
                ok = synthesize_edge_tts(text, voice, output_path)
            elif backend == "fish":
                voice_id = voice.stem
                filename = make_output_filename(text, voice_id, idx)
                output_path = out / filename
                ok = synthesize_fish_speech(text, voice, output_path, api_url, lang)
            else:  # xtts
                voice_id = voice.stem
                filename = make_output_filename(text, voice_id, idx)
                output_path = out / filename
                ok = synthesize_xtts(text, voice, output_path, lang)

            if ok and output_path.exists():
                info = sf.info(str(output_path))
                manifest.append({
                    "audio_filepath": str(output_path),
                    "text": text,
                    "duration": info.duration,
                    "language": item.get("language", lang),
                    "source": "synthetic",
                    "voice_id": voice_id,
                    "augmentation": None,
                })

    # Write manifest
    manifest_path = out / "manifest_clean.jsonl"
    with open(manifest_path, "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")

    print(f"Synthesized {len(manifest)} audio files. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
