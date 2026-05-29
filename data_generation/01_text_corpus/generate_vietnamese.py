"""Generate Vietnamese text corpus using OpenAI API, grounded in real VIVOS transcripts."""
import csv
import json
import random
from pathlib import Path

import click
from tqdm import tqdm

try:
    import openai
except ImportError:
    raise ImportError("pip install openai")

from dotenv import load_dotenv
load_dotenv()  # loads OPENAI_API_KEY from .env if present

MODEL = "gpt-4.1"
# Real Vietnamese transcripts from InfoRe Technology (CC-BY-4.0, 25 hrs).
# Parquet-based, loads without audio decoding issues.
# Used as few-shot grounding so GPT doesn't hallucinate diacritics or phrasing.
REFERENCE_DATASET = "doof-ferb/infore1_25hours"
REFERENCE_MAX_SAMPLES = 3000


def load_reference_sentences(dataset_id: str, max_samples: int) -> list[str]:
    """Load real Vietnamese sentences from a HuggingFace dataset.

    Falls back gracefully to an empty list if the datasets library is not
    installed or the dataset cannot be fetched (e.g. offline).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[WARN] 'datasets' not installed — running without reference grounding.")
        print("       pip install -e '.[prototype]'")
        return []

    print(f"Loading reference dataset: {dataset_id} (streaming, text only) ...")
    try:
        ds = load_dataset(dataset_id, split="train", streaming=True)
    except Exception as e:
        print(f"[WARN] Could not load {dataset_id}: {e}")
        print("       Continuing without reference grounding.")
        return []

    # Find the text column, then select only that column to avoid audio decoding
    text_col = None
    for col in ("sentence", "transcription", "text", "transcript", "normalized_text"):
        if col in ds.column_names:
            text_col = col
            break

    if text_col is None:
        print(f"[WARN] No recognised text column in {dataset_id}. Columns: {ds.column_names}")
        return []

    ds = ds.select_columns([text_col])  # drop audio column — avoids torchcodec dependency

    sentences = []
    for row in ds:
        val = row[text_col]
        if isinstance(val, str) and len(val.strip()) > 8:
            sentences.append(val)
        if len(sentences) >= max_samples:
            break

    print(f"  Loaded {len(sentences)} reference sentences (column: '{text_col}')")
    return sentences


def load_lexicon(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def generate_batch(
    client: openai.OpenAI,
    terms: list[str],
    system_prompt: str,
    reference_pool: list[str],
    batch_size: int = 20,
) -> list[str]:
    # Sample a few real sentences to anchor the model's diacritics, register and phrasing
    few_shot_block = ""
    if reference_pool:
        examples = random.sample(reference_pool, k=min(6, len(reference_pool)))
        few_shot_block = (
            "Here are real Vietnamese sentences recorded from native speakers. "
            "Use these as a style and diacritics guide — do NOT copy them, generate new ones:\n"
            + "\n".join(f"- {s}" for s in examples)
            + "\n\n"
        )

    user_msg = (
        f"{few_shot_block}"
        f"Generate {batch_size} NEW Vietnamese sentences. "
        f"Each must naturally include at least one of these terms: {', '.join(terms)}\n\n"
        f"Return ONLY a JSON array of strings. No markdown, no explanation."
    )

    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=4000,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
    )

    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        sentences = json.loads(text)
        if isinstance(sentences, list):
            return [s for s in sentences if isinstance(s, str) and len(s.strip()) > 0]
    except json.JSONDecodeError:
        print(f"[WARN] Failed to parse batch, skipping. Raw: {text[:200]}")
    return []


@click.command()
@click.option("--output", "-o", required=True, help="Output JSONL path")
@click.option("--count", "-n", default=1000, help="Target sentence count")
@click.option("--batch-size", default=20, help="Sentences per API call")
@click.option(
    "--lexicon",
    default="data_generation/01_text_corpus/lexicons/vietnamese_tonal.csv",
    help="Lexicon CSV path",
)
@click.option(
    "--skip-reference",
    is_flag=True,
    default=False,
    help="Skip loading the HuggingFace reference dataset (offline mode)",
)
def main(output: str, count: int, batch_size: int, lexicon: str, skip_reference: bool):
    client = openai.OpenAI()  # uses OPENAI_API_KEY env var

    lex = load_lexicon(lexicon)
    terms = [row["term"] for row in lex]

    system_prompt = Path("data_generation/01_text_corpus/prompts/vietnamese_system.txt").read_text()

    reference_pool: list[str] = []
    if not skip_reference:
        reference_pool = load_reference_sentences(REFERENCE_DATASET, REFERENCE_MAX_SAMPLES)
        if not reference_pool:
            print("  Proceeding without reference grounding (generation still works).")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_sentences = []
    n_calls = (count // batch_size) + 1

    print(f"Generating ~{count} Vietnamese sentences in {n_calls} API calls...")

    for _ in tqdm(range(n_calls)):
        batch_terms = random.sample(terms, k=min(random.randint(4, 7), len(terms)))
        sentences = generate_batch(client, batch_terms, system_prompt, reference_pool, batch_size)
        all_sentences.extend(sentences)

        if len(all_sentences) >= count:
            break

    # Deduplicate, preserving order
    all_sentences = list(dict.fromkeys(all_sentences))[:count]

    with open(output_path, "w") as f:
        for sent in all_sentences:
            f.write(json.dumps({"text": sent, "language": "vi"}) + "\n")

    print(f"Wrote {len(all_sentences)} sentences to {output_path}")


if __name__ == "__main__":
    main()
