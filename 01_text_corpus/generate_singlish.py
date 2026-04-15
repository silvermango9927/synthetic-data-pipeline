"""Generate Singlish text corpus using Claude API."""
import csv
import json
import random
from pathlib import Path

import click
from tqdm import tqdm

try:
    import anthropic
except ImportError:
    raise ImportError("pip install anthropic")

MODEL = "claude-sonnet-4-6"


def load_lexicon(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def chunk_list(lst: list, n: int) -> list[list]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def generate_batch(
    client: anthropic.Anthropic,
    terms: list[str],
    system_prompt: str,
    batch_size: int = 20,
) -> list[str]:
    user_msg = (
        f"Generate {batch_size} Singlish sentences. "
        f"Each must include at least one of these terms: {', '.join(terms)}\n\n"
        f"Return ONLY a JSON array of strings."
    )

    resp = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}],
    )

    text = resp.content[0].text.strip()
    # Strip markdown fences if present
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
    default="01_text_corpus/lexicons/singlish_particles.csv",
    help="Lexicon CSV path",
)
def main(output: str, count: int, batch_size: int, lexicon: str):
    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

    lex = load_lexicon(lexicon)
    terms = [row["term"] for row in lex]

    system_prompt = Path("01_text_corpus/prompts/singlish_system.txt").read_text()

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_sentences = []
    n_calls = (count // batch_size) + 1

    print(f"Generating ~{count} Singlish sentences in {n_calls} API calls...")

    for _ in tqdm(range(n_calls)):
        # Sample 5-8 terms per batch for variety
        batch_terms = random.sample(terms, k=min(random.randint(5, 8), len(terms)))
        sentences = generate_batch(client, batch_terms, system_prompt, batch_size)
        all_sentences.extend(sentences)

        if len(all_sentences) >= count:
            break

    # Deduplicate
    all_sentences = list(dict.fromkeys(all_sentences))[:count]

    # Write JSONL
    with open(output_path, "w") as f:
        for sent in all_sentences:
            f.write(json.dumps({"text": sent, "language": "en-SG"}) + "\n")

    print(f"Wrote {len(all_sentences)} sentences to {output_path}")


if __name__ == "__main__":
    main()
