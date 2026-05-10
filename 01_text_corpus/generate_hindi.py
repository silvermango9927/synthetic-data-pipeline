"""Generate Hindi text corpus using OpenAI API, grounded in real FLEURS hi_in transcripts.

Two length targets are supported via --length-target:
  short  (8–15 words / ~5s spoken)
  long   (75–110 words / ~30s spoken)

Each target uses its own system prompt under prompts/hindi_{short,long}_system.txt.
"""
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
load_dotenv()

MODEL = "gpt-4.1"
# google/fleurs hi_in — Hindi read speech, CC-BY-4.0, ungated, ~3k sentences.
# FLEURS sentences are pure Hindi (no code-switching); we instruct the LLM
# to add Hinglish code-switching naturally in the prompt.
REFERENCE_DATASET = "google/fleurs"
REFERENCE_CONFIG = "hi_in"
REFERENCE_MAX_SAMPLES = 3000


def load_reference_sentences(
    dataset_id: str, config: str | None, max_samples: int
) -> list[str]:
    """Load real Hindi sentences from a HuggingFace dataset.

    Falls back gracefully to an empty list if the datasets library is not
    installed or the dataset cannot be fetched (e.g. offline).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[WARN] 'datasets' not installed — running without reference grounding.")
        print("       pip install -e '.[prototype]'")
        return []

    print(f"Loading reference dataset: {dataset_id} ({config}) (streaming, text only) ...")
    try:
        if config:
            ds = load_dataset(dataset_id, config, split="train", streaming=True)
        else:
            ds = load_dataset(dataset_id, split="train", streaming=True)
    except Exception as e:
        print(f"[WARN] Could not load {dataset_id}: {e}")
        print("       Continuing without reference grounding.")
        return []

    text_col = None
    for col in ("transcription", "raw_transcription", "sentence", "text", "transcript"):
        if col in ds.column_names:
            text_col = col
            break

    if text_col is None:
        print(f"[WARN] No recognised text column in {dataset_id}. Columns: {ds.column_names}")
        return []

    ds = ds.select_columns([text_col])

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
    batch_size: int,
    length_target: str,
) -> list[str]:
    few_shot_block = ""
    if reference_pool:
        examples = random.sample(reference_pool, k=min(6, len(reference_pool)))
        few_shot_block = (
            "Here are real Hindi sentences from native Hindi speakers (FLEURS). "
            "These are pure Hindi — your output should follow this Devanagari style "
            "but ALSO add natural Hinglish code-switching (mix in 1–3 English words). "
            "Do NOT copy them, generate new ones:\n"
            + "\n".join(f"- {s}" for s in examples)
            + "\n\n"
        )

    if length_target == "short":
        ask = (
            f"Generate {batch_size} NEW Hindi sentences (Devanagari, 8–15 words each, "
            f"~5 seconds spoken)."
        )
    else:
        ask = (
            f"Generate {batch_size} NEW Hindi passages (Devanagari, 75–110 words each, "
            f"~30 seconds spoken). Each passage is one coherent monologue."
        )

    user_msg = (
        f"{few_shot_block}"
        f"{ask} Each must naturally include at least one of these terms: {', '.join(terms)}\n\n"
        f"Return ONLY a JSON array of strings. No markdown, no explanation."
    )

    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=4000 if length_target == "short" else 8000,
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
@click.option("--batch-size", default=10, help="Sentences per API call")
@click.option(
    "--lexicon",
    default="01_text_corpus/lexicons/hindi_common.csv",
    help="Lexicon CSV path",
)
@click.option(
    "--length-target",
    type=click.Choice(["short", "long"]),
    default="short",
    help="short = 8–15 words (~5s), long = 75–110 words (~30s)",
)
@click.option(
    "--skip-reference",
    is_flag=True,
    default=False,
    help="Skip loading the HuggingFace reference dataset (offline mode)",
)
def main(
    output: str,
    count: int,
    batch_size: int,
    lexicon: str,
    length_target: str,
    skip_reference: bool,
):
    client = openai.OpenAI()  # uses OPENAI_API_KEY env var

    lex = load_lexicon(lexicon)
    terms = [row["term"] for row in lex]

    prompt_path = Path(f"01_text_corpus/prompts/hindi_{length_target}_system.txt")
    system_prompt = prompt_path.read_text()

    reference_pool: list[str] = []
    if not skip_reference:
        reference_pool = load_reference_sentences(
            REFERENCE_DATASET, REFERENCE_CONFIG, REFERENCE_MAX_SAMPLES
        )
        if not reference_pool:
            print("  Proceeding without reference grounding (generation still works).")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_sentences = []
    n_calls = (count // batch_size) + 1

    print(
        f"Generating ~{count} Hindi sentences ({length_target}-form) in {n_calls} API calls..."
    )

    for _ in tqdm(range(n_calls)):
        batch_terms = random.sample(terms, k=min(random.randint(4, 7), len(terms)))
        sentences = generate_batch(
            client, batch_terms, system_prompt, reference_pool, batch_size, length_target
        )
        all_sentences.extend(sentences)

        if len(all_sentences) >= count:
            break

    all_sentences = list(dict.fromkeys(all_sentences))[:count]

    with open(output_path, "w") as f:
        for sent in all_sentences:
            f.write(json.dumps({"text": sent, "language": "hi", "length_target": length_target}) + "\n")

    print(f"Wrote {len(all_sentences)} sentences to {output_path}")


if __name__ == "__main__":
    main()
