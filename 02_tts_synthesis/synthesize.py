"""Synthesize audio from text corpus using TTS."""
import asyncio
import base64
import hashlib
import io
import json
import os
import random
from pathlib import Path

import click
import soundfile as sf
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()  # picks up DASHSCOPE_API_KEY / SARVAM_API_KEY from .env

# Default edge-tts voices per language (Microsoft Neural TTS, free, no server needed)
EDGE_VOICES = {
    "en": ["en-SG-LunaNeural", "en-SG-WayneNeural", "en-US-JennyNeural", "en-US-GuyNeural"],
    "vi": ["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"],
    "zh": [
        "zh-CN-XiaoxiaoNeural",
        "zh-CN-YunyangNeural",
        "zh-CN-XiaoyiNeural",
        "zh-CN-YunxiNeural",
        "zh-CN-liaoning-XiaobeiNeural",
        "zh-CN-shaanxi-XiaoniNeural",
    ],
    "hi": ["hi-IN-SwaraNeural", "hi-IN-MadhurNeural"],
}

# Qwen3-TTS-Flash preset voices (DashScope) — mixed gender + regional flavour
QWEN_VOICES = ["Cherry", "Serena", "Ethan", "Chelsie", "Vivian", "Momo"]

# Sarvam bulbul:v3 Hindi-capable speakers — mixed gender (validated against the model's whitelist)
SARVAM_VOICES = ["shubh", "aditya", "priya", "neha", "ritu", "rohan"]

# MiniMax T2A v2 (speech-02-hd) Chinese system voices — mixed gender + register
MINIMAX_VOICES = [
    "Chinese (Mandarin)_News_Anchor",
    "Chinese (Mandarin)_Reliable_Executive",
    "Chinese (Mandarin)_Mature_Woman",
    "Chinese (Mandarin)_Warm_Girl",
    "Chinese (Mandarin)_Unrestrained_Young_Man",
    "Chinese (Mandarin)_Crisp_Girl",
]

# Endpoint for Alibaba Cloud Model Studio (international / Singapore region).
# Mainland keys use https://dashscope.aliyuncs.com/api/v1 instead.
DASHSCOPE_INTL_BASE_URL = "https://dashscope-intl.aliyuncs.com/api/v1"

# MiniMax T2A v2 endpoint (international). Mainland endpoint is api.minimaxi.com.
MINIMAX_API_URL = "https://api.minimax.io/v1/t2a_v2"


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


def synthesize_qwen_tts(
    text: str,
    voice: str,
    output_path: Path,
    model: str = "qwen3-tts-flash",
) -> bool:
    """Synthesize Mandarin via Qwen TTS on Alibaba DashScope (international/Singapore region).

    Returns 16 kHz mono WAV (downmixed + resampled from native 24 kHz).
    Requires DASHSCOPE_API_KEY in env AND the qwen3-tts-flash model to be activated
    in your Alibaba Cloud Model Studio account (free tier; one-click in console).
    Install: pip install -e '.[tts-cloud]'
    """
    try:
        import dashscope
        from dashscope.audio.qwen_tts import SpeechSynthesizer
    except ImportError:
        raise ImportError("pip install -e '.[tts-cloud]'  # for Qwen TTS")

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("[ERROR] DASHSCOPE_API_KEY not set in env / .env")
        return False
    dashscope.api_key = api_key
    # International (Singapore) endpoint — keys provisioned outside mainland China use this.
    dashscope.base_http_api_url = DASHSCOPE_INTL_BASE_URL

    try:
        resp = SpeechSynthesizer.call(model=model, text=text, voice=voice)
    except Exception as e:
        print(f"[ERROR] Qwen TTS call raised: {e}")
        return False

    if getattr(resp, "status_code", None) != 200:
        msg = getattr(resp, "message", "?")
        code = getattr(resp, "code", "?")
        print(f"[ERROR] Qwen TTS {code}: {msg}")
        if code == "AccessDenied.Unpurchased":
            print("        Activate qwen3-tts-flash in Alibaba Cloud Model Studio:")
            print("        https://modelstudio.console.alibabacloud.com (Singapore region)")
        return False

    # Successful response: resp.output.audio.{data (b64) | url}
    audio = getattr(resp.output, "audio", None) if resp.output else None
    if audio is None:
        print(f"[ERROR] Qwen TTS missing output.audio: {resp}")
        return False

    audio_bytes = None
    data_b64 = audio.get("data") if isinstance(audio, dict) else getattr(audio, "data", None)
    audio_url = audio.get("url") if isinstance(audio, dict) else getattr(audio, "url", None)
    if data_b64:
        try:
            audio_bytes = base64.b64decode(data_b64)
        except Exception as e:
            print(f"[ERROR] Qwen TTS b64 decode failed: {e}")
            return False
    elif audio_url:
        try:
            import httpx
            audio_bytes = httpx.get(audio_url, timeout=60.0).content
        except Exception as e:
            print(f"[ERROR] Qwen TTS audio URL fetch failed: {e}")
            return False

    if not audio_bytes:
        print(f"[ERROR] Qwen TTS returned no audio payload. Response: {resp}")
        return False

    try:
        import librosa

        audio_arr, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if audio_arr.ndim > 1:
            audio_arr = audio_arr.mean(axis=1)
        if sr != 16000:
            audio_arr = librosa.resample(audio_arr, orig_sr=sr, target_sr=16000)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio_arr, 16000)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to write Qwen TTS WAV: {e}")
        return False


def synthesize_minimax_tts(
    text: str,
    voice: str,
    output_path: Path,
    model: str = "speech-02-hd",
    language_boost: str = "Chinese",
    sample_rate: int = 16000,
) -> bool:
    """Synthesize Mandarin via MiniMax T2A v2 (speech-02-hd).

    Native 16 kHz WAV output (no resample needed). Hex-encoded audio in response body.
    Requires MINIMAX_API_KEY in env.
    """
    import httpx

    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        print("[ERROR] MINIMAX_API_KEY not set in env / .env")
        return False

    payload = {
        "model": model,
        "text": text,
        "stream": False,
        "language_boost": language_boost,
        "output_format": "hex",
        "voice_setting": {"voice_id": voice, "speed": 1, "vol": 1, "pitch": 0},
        "audio_setting": {
            "sample_rate": sample_rate,
            "bitrate": 128000,
            "format": "wav",
            "channel": 1,
        },
    }

    try:
        resp = httpx.post(
            MINIMAX_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120.0,
        )
    except httpx.HTTPError as e:
        print(f"[ERROR] MiniMax request failed: {e}")
        return False

    if resp.status_code != 200:
        print(f"[ERROR] MiniMax returned {resp.status_code}: {resp.text[:300]}")
        return False

    try:
        body = resp.json()
        # MiniMax wraps errors inside a 200 with base_resp.status_code != 0
        base = body.get("base_resp", {})
        if base.get("status_code", 0) != 0:
            print(f"[ERROR] MiniMax base_resp {base.get('status_code')}: {base.get('status_msg')}")
            return False
        audio_hex = body.get("data", {}).get("audio")
        if not audio_hex:
            print(f"[ERROR] MiniMax response missing data.audio: {str(body)[:200]}")
            return False
        audio_bytes = bytes.fromhex(audio_hex)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(audio_bytes)
        # Sanity-correct to 16 kHz mono if MiniMax echoed a different config back.
        info = sf.info(str(output_path))
        if info.samplerate != 16000 or info.channels != 1:
            audio, sr = sf.read(str(output_path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sf.write(str(output_path), audio, 16000)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to decode/write MiniMax WAV: {e}")
        return False


def synthesize_sarvam_tts(
    text: str,
    voice: str,
    output_path: Path,
    model: str = "bulbul:v3",
    target_language_code: str = "hi-IN",
) -> bool:
    """Synthesize Hindi via Sarvam bulbul:v3.

    Returns 16 kHz mono WAV (Sarvam supports 16 kHz natively — no resample).
    Requires SARVAM_API_KEY in env. Install: pip install -e '.[tts-cloud]'
    Sarvam bulbul:v3 max chars per call = 2500.
    """
    import httpx

    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        print("[ERROR] SARVAM_API_KEY not set in env / .env")
        return False

    if len(text) > 2500:
        print(f"[WARN] Sarvam input >2500 chars ({len(text)}); truncating.")
        text = text[:2500]

    try:
        resp = httpx.post(
            "https://api.sarvam.ai/text-to-speech",
            headers={"api-subscription-key": api_key, "Content-Type": "application/json"},
            json={
                "text": text,
                "target_language_code": target_language_code,
                "speaker": voice,
                "model": model,
                "speech_sample_rate": 16000,
            },
            timeout=120.0,
        )
    except httpx.HTTPError as e:
        print(f"[ERROR] Sarvam request failed: {e}")
        return False

    if resp.status_code != 200:
        print(f"[ERROR] Sarvam returned {resp.status_code}: {resp.text[:200]}")
        return False

    try:
        body = resp.json()
        audios = body.get("audios") or [body.get("audio")]
        b64 = audios[0] if audios else None
        if not b64:
            print(f"[ERROR] Sarvam response missing 'audios': {str(body)[:200]}")
            return False
        audio_bytes = base64.b64decode(b64)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Sarvam returns WAV bytes already at 16kHz when speech_sample_rate=16000
        output_path.write_bytes(audio_bytes)
        # Sanity: confirm 16kHz mono — re-read and rewrite if not.
        info = sf.info(str(output_path))
        if info.samplerate != 16000 or info.channels != 1:
            audio, sr = sf.read(str(output_path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sf.write(str(output_path), audio, 16000)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to decode/write Sarvam WAV: {e}")
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
    type=click.Choice(["fish", "xtts", "edge", "qwen", "minimax", "sarvam"]),
    help=(
        "TTS backend. 'edge' = Microsoft Neural (en/vi/zh/hi free fallback); "
        "'qwen' = DashScope Qwen TTS (zh, requires activation); "
        "'minimax' = MiniMax speech-02-hd (zh, primary); "
        "'sarvam' = Sarvam bulbul:v3 (hi, primary). 'fish'/'xtts' need a voice bank."
    ),
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
    elif backend == "qwen":
        voices = list(QWEN_VOICES)
        print(f"qwen TTS backend: using {len(voices)} preset voices")
    elif backend == "minimax":
        voices = list(MINIMAX_VOICES)
        print(f"minimax TTS backend (speech-02-hd): using {len(voices)} preset Chinese voices")
    elif backend == "sarvam":
        voices = list(SARVAM_VOICES)
        print(f"sarvam TTS backend: using {len(voices)} preset voices")
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
            elif backend == "qwen":
                voice_id = f"qwen_{voice}"
                filename = make_output_filename(text, voice_id, idx)
                output_path = out / filename
                ok = synthesize_qwen_tts(text, voice, output_path)
            elif backend == "minimax":
                # Sanitize "Chinese (Mandarin)_News_Anchor" → "minimax_News_Anchor"
                short_voice = voice.split("_", 1)[-1] if "_" in voice else voice
                voice_id = f"minimax_{short_voice}".replace(" ", "_").replace("(", "").replace(")", "")
                filename = make_output_filename(text, voice_id, idx)
                output_path = out / filename
                ok = synthesize_minimax_tts(text, voice, output_path)
            elif backend == "sarvam":
                voice_id = f"sarvam_{voice}"
                filename = make_output_filename(text, voice_id, idx)
                output_path = out / filename
                lang_code = "hi-IN" if lang.startswith("hi") else lang
                ok = synthesize_sarvam_tts(
                    text, voice, output_path, target_language_code=lang_code
                )
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
