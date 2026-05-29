"""Preflight environment check for the VALSEA ASR data generation pipeline."""
import importlib
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads OPENAI_API_KEY from .env if present
except ImportError:
    pass

GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


def ok(label: str, detail: str = "") -> None:
    line = f"  {GREEN}[OK]{RESET}   {label}"
    if detail:
        line += f"  — {detail}"
    print(line)


def warn(label: str, detail: str = "") -> None:
    line = f"  {YELLOW}[WARN]{RESET} {label}"
    if detail:
        line += f"  — {detail}"
    print(line)


def fail(label: str, detail: str = "") -> None:
    line = f"  {RED}[FAIL]{RESET} {label}"
    if detail:
        line += f"  — {detail}"
    print(line)


def main() -> int:
    print("\n=== VALSEA ASR Pipeline — Environment Check ===\n")
    hard_failures = 0

    # ── Core packages ──────────────────────────────────────────────────────────
    print("Core packages (required for all targets):")
    core_pkgs = ["openai", "click", "tqdm", "soundfile", "numpy", "pydantic", "librosa"]
    for pkg in core_pkgs:
        try:
            importlib.import_module(pkg)
            ok(pkg)
        except ImportError:
            fail(pkg, f"pip install -e '.[prototype,dev]'")
            hard_failures += 1

    # ── Prototype packages ─────────────────────────────────────────────────────
    print("\nPrototype packages (required for make prototype-edge):")
    try:
        importlib.import_module("edge_tts")
        ok("edge_tts")
    except ImportError:
        fail("edge_tts", "pip install -e '.[prototype]'")
        hard_failures += 1

    # audiomentations is optional — augment.py degrades gracefully without it
    try:
        importlib.import_module("audiomentations")
        ok("audiomentations", "full augmentation enabled")
    except ImportError:
        warn(
            "audiomentations",
            "not installed — augment.py will copy files without transforms "
            "(to enable: sudo xcodebuild -license && pip install -e '.[augment]')",
        )

    # ── API key ────────────────────────────────────────────────────────────────
    print("\nAPI keys:")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        ok("OPENAI_API_KEY", f"set ({api_key[:10]}...)")
    else:
        fail("OPENAI_API_KEY", "export OPENAI_API_KEY=sk-...")
        hard_failures += 1

    # ── Directory structure ────────────────────────────────────────────────────
    print("\nDirectory structure:")
    required_dirs = [
        "data_generation/01_text_corpus/lexicons",
        "data_generation/01_text_corpus/prompts",
        "data_generation/02_tts_synthesis/voice_bank/singlish",
        "data_generation/02_tts_synthesis/voice_bank/vietnamese",
        "data_generation/03_augmentation/noise_bank/ambient",
        "data_generation/03_augmentation/noise_bank/rir",
        "outputs/singlish/clean",
        "outputs/singlish/augmented",
        "outputs/vietnamese/clean",
        "outputs/vietnamese/augmented",
    ]
    for d in required_dirs:
        if Path(d).is_dir():
            ok(d)
        else:
            fail(d, "missing — re-run scaffold")
            hard_failures += 1

    # ── Voice bank (soft — only needed for fish/xtts) ─────────────────────────
    print("\nVoice bank (only needed for --backend fish or xtts):")
    for lang in ("singlish", "vietnamese"):
        vb = Path(f"data_generation/02_tts_synthesis/voice_bank/{lang}")
        wavs = list(vb.glob("*.wav")) + list(vb.glob("*.mp3")) + list(vb.glob("*.flac"))
        if wavs:
            ok(f"voice_bank/{lang}", f"{len(wavs)} file(s)")
        else:
            warn(f"voice_bank/{lang}", "empty — OK for --backend edge, needed for fish/xtts")

    # ── Noise bank (soft — augment.py degrades gracefully) ────────────────────
    print("\nNoise bank (optional — augment.py degrades gracefully without it):")
    for subdir in ("ambient", "rir"):
        nb = Path(f"data_generation/03_augmentation/noise_bank/{subdir}")
        wavs = list(nb.glob("*.wav"))
        if wavs:
            ok(f"noise_bank/{subdir}", f"{len(wavs)} file(s)")
        else:
            warn(f"noise_bank/{subdir}", "empty — run scripts/download_noise.sh to populate")

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    if hard_failures == 0:
        print(f"{GREEN}All checks passed.{RESET} Run: bash scripts/run_prototype.sh")
    else:
        print(f"{RED}{hard_failures} blocker(s) found.{RESET} Fix the FAILs above, then retry.")
        print()
        print("Quick fix:")
        print("  pip install -e '.[prototype,dev]'")
        print("  export OPENAI_API_KEY=sk-...")
        print("  make prototype-edge")

    print()
    return 1 if hard_failures > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
