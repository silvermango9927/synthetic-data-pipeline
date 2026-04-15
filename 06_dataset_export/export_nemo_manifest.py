"""Export filtered data to NeMo-compatible manifest format."""
import json
from pathlib import Path

import click


@click.command()
@click.option("--input", "input_path", required=True, help="Filtered manifest JSONL")
@click.option("--output", "output_path", required=True, help="Output NeMo manifest JSON")
@click.option("--audio-base", default=None, help="Rebase audio paths relative to this dir")
def main(input_path: str, output_path: str, audio_base: str | None):
    """Convert internal manifest to NeMo training format.

    NeMo format: one JSON object per line with keys:
    - audio_filepath (relative or absolute path to WAV)
    - text (ground truth transcript)
    - duration (float, seconds)
    """
    entries = []
    with open(input_path) as f:
        for line in f:
            entry = json.loads(line)

            nemo_entry = {
                "audio_filepath": entry["audio_filepath"],
                "text": entry["text"],
                "duration": entry["duration"],
            }

            if audio_base:
                nemo_entry["audio_filepath"] = str(
                    Path(entry["audio_filepath"]).relative_to(audio_base)
                )

            entries.append(nemo_entry)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    total_hours = sum(e["duration"] for e in entries) / 3600
    print(f"Exported {len(entries)} samples ({total_hours:.1f} hours) to {out}")


if __name__ == "__main__":
    main()
