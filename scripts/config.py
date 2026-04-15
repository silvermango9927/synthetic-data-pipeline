"""Shared configuration and data models for the pipeline."""
from enum import Enum
from pathlib import Path

from pydantic import BaseModel


class Language(str, Enum):
    SINGLISH = "en"
    VIETNAMESE = "vi"


class SampleMetadata(BaseModel):
    """Metadata for a single audio-text pair."""

    audio_filepath: str
    text: str
    duration: float
    language: str
    source: str  # "synthetic" or "real"
    voice_id: str | None = None
    augmentation: str | None = None  # description of augmentation applied
    utmos_score: float | None = None
    roundtrip_wer: float | None = None


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    # TTS
    tts_api_url: str = "http://localhost:8080/v1/tts"  # Fish Speech local server
    voices_per_sentence: int = 3

    # Quality thresholds
    utmos_threshold: float = 3.5
    wer_threshold_vi: float = 0.15
    wer_threshold_en: float = 0.25

    # Augmentation
    augmentation_variants: int = 2  # augmented copies per clean sample
    snr_range: tuple[float, float] = (10.0, 25.0)

    # Paths
    noise_bank: Path = Path("03_augmentation/noise_bank")
    output_base: Path = Path("outputs")


# Singleton config, override with env vars or CLI args
CONFIG = PipelineConfig()
