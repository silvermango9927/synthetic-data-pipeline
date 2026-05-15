# Bulk-production readiness report

Generated against branch `add-bulk-readiness` on 2026-05-15 from a default-mode smoke run
(`bash scripts/smoke_test_bulk.sh`, logs in [logs/smoke_20260515_140841/](logs/smoke_20260515_140841/)).

The smoke test exited `0` with PASS on every verification stage — **but the test silently
passed through several real production blockers**. The PASS-but-broken behavior is itself a
finding (P0).

> **Status update (2026-05-15, post-fix):** All P0 and P1 items resolved. Hindi default
> switched to edge-tts. Re-run smoke test
> ([logs/smoke_20260515_173651/](logs/smoke_20260515_173651/)) PASSES cleanly with the new
> dedup assertion in place — `train_manifest.json` rows are now 100% unique across all 4
> (lang, bucket) pairs. Polluted prototype outputs preserved at
> `outputs.polluted_20260515_173520/` for review/deletion. Per-section "✅ fixed" markers
> below describe what landed.

---

## TL;DR

| # | Severity | Issue | Blocks bulk? | Status |
|---|---|---|---|---|
| 1 | **P0** | Sarvam API has **0 credits** — every Hindi TTS call 429s with `insufficient_quota_error`. | **Yes** | ✅ Fixed — Hindi default switched to edge-tts. Sarvam stays opt-in via `TTS_BACKEND_HI=sarvam`, and synthesize.py now aborts after 10 consecutive 429s instead of silently producing nothing. |
| 2 | **P0** | [04_quality_filter/filter.py:179](04_quality_filter/filter.py#L179) `rglob("manifest_*.jsonl")` re-ingests its own previous output AND `augmented_real/`. NeMo `train_manifest.json` already has **1.54× – 1.91× duplicate rows**. | **Yes** | ✅ Fixed — explicit `clean/manifest_clean.jsonl` + `augmented/manifest_augmented.jsonl` listing. Re-run smoke shows `158 rows (158 unique)` across all 4 (lang, bucket) pairs. |
| 3 | **P0** | Voice-rotation switch from `random.sample` → `rotate_voices` changed filenames (filename embeds `voice_id`). The smoke run, in default mode, generated **66 new zh/short and 65 new zh/long edge-tts WAVs** on top of the existing 100 each, because resume couldn't match old filenames. | **Yes** | ✅ Fixed — polluted clean/aug dirs moved to `outputs.polluted_20260515_173520/`; bulk will regenerate uniformly under `rotate_voices`. Edge-tts is free so regen has no $ cost. |
| 4 | **P0** | Smoke test does **not fail** on Sarvam 429s, on voice-rotation pollution, or on duplicate-row filter output. | **Yes** | ✅ Fixed — smoke now asserts no duplicate `audio_filepath` in `train_manifest.json`; synthesize.py raises `SystemExit` on consecutive-failure abort, which propagates through the smoke script's `set -euo pipefail`. |
| 5 | **P1** | [pyproject.toml:21](pyproject.toml#L21) pins `audiomentations>=0.36.0` which is broken on Apple Silicon. | No | ✅ Fixed — pin is now `>=0.35,<0.36` and `pydub` is in the same extras block. `augment.py` error message updated to match. |
| 6 | **P1** | [03_augmentation/augment.py](03_augmentation/augment.py) has **no resume** — every bulk re-run re-augments every clean WAV. | No | ✅ Fixed — augment.py now skips wavs whose `_aug{v:02d}.wav` already exists and is readable; manifest row is reconstructed from disk. Prints `(N new, K resumed)` summary. |
| 7 | **P2** | Stale legacy outputs (`outputs/{chinese,hindi}/{short}/augmented_real/`) sit inside the language tree and were pulled in by the filter `rglob`. | No | ✅ Fixed (by P0-2 + cleanup) — filter no longer rglobs, and the legacy dirs are now in `outputs.polluted_*/`. |
| 8 | **P2** | `BUILDPLAN.md` stale; README's Quick Start install string is heavier than necessary. | No | ⚠️ Partial — README install string lightened, CLAUDE.md + BULK_GENERATION_PLAN.md updated to reflect edge-tts-default Hindi. `BUILDPLAN.md` left as historical (April vintage); deletion or refresh deferred. |
| 9 | **P2** | [Makefile:18-32](Makefile#L18-L32) recipes missing space between `$(PYTHON)` and the script path. | No | ⏸ Not changed — Makefile already documented as broken in CLAUDE.md; bulk path doesn't use `make`. |
| 10 | **P3** | `synthesize.py` truncates Sarvam input >2,500 chars **silently**. | No | ⏸ Not changed — unchanged risk (only triggers on Sarvam opt-in with ≥2,500-char input, which is unlikely at our text-gen targets). |
| 11 | **P3** | `tests/` directory is empty. | No | ⏸ Not changed — smoke test is now the verification harness. |

---

## What the smoke test actually exercised

Run command: `bash scripts/smoke_test_bulk.sh` (default mode, against existing outputs).

### Pre-run state (relevant)

```
outputs/chinese/short/clean: 100 wavs   (50 sentences × 2 voices, random.sample rotation)
outputs/chinese/long/clean:  100 wavs
outputs/hindi/short/clean:   100 wavs
outputs/hindi/long/clean:     67 wavs   (Sarvam quit mid-prototype run)
```

### Post-run state

```
outputs/chinese/short/clean: 166 wavs   (+66 new from edge-tts under new rotation)
outputs/chinese/long/clean:  165 wavs   (+65 new)
outputs/hindi/short/clean:   100 wavs   (unchanged — all 66 new Sarvam jobs 429'd)
outputs/hindi/long/clean:     67 wavs   (all 78 new Sarvam jobs 429'd)
```

### Verification output (every line is PASS)

```
[3/7] Verify: clean WAV counts per (lang, bucket)
  PASS chinese/short: 166 clean WAVs (>= 40)
  PASS chinese/long:  165 clean WAVs (>= 10)
  PASS hindi/short:   100 clean WAVs (>= 40)
  PASS hindi/long:     67 clean WAVs (>= 10)

[4/7] Verify: augmented manifests have real transforms (not passthrough)
  PASS chinese/short: 166 aug rows, 0 passthrough
  PASS chinese/long:  165 aug rows, 0 passthrough
  PASS hindi/short:   100 aug rows, 0 passthrough
  PASS hindi/long:     67 aug rows, 0 passthrough

[5/7] Verify: train_manifest.json is valid NeMo
  PASS chinese/short: 409 rows, 27.9  min audio  ← 1.54x duplicates
  PASS chinese/long:  530 rows, 259.8 min audio  ← 1.61x duplicates
  PASS hindi/short:   420 rows, 26.8  min audio  ← 1.91x duplicates
  PASS hindi/long:    268 rows, 141.9 min audio

[6/7] HF staging + load_dataset accepts the schema   PASS (all 8 configs)
[7/7] Smoke test PASSED
```

The PASS lines hide all four P0 issues. Detail below.

---

## P0-1 — Sarvam credits exhausted

**Evidence:** `logs/smoke_20260515_140841/02_bulk.log` contains 100+ lines of:
```
[ERROR] Sarvam returned 429: {"error":{"message":"No credits available. Please visit
API dashboard to buy more credits.","code":"insufficient_quota_error", ...}}
```
followed by `rate_limit_exceeded_error`. Result: `Synthesized 0 new audio files` for `hi/short`,
and a similar zero-add for `hi/long`.

**Why bulk fails:** [scripts/run_bulk.sh:117-118](scripts/run_bulk.sh#L117-L118) defaults
`TTS_BACKEND_HI=sarvam` and a 24-hr Hindi run needs ~7,000 new Sarvam calls (~540k chars).
With 0 credits, every call 429s and the run silently produces no new Hindi audio.

**Fix options (pick one before launch):**
- Top up Sarvam to cover ~540k chars (per BULK_GENERATION_PLAN.md, $5–$30 plan-dependent).
- Or run with `TTS_BACKEND_HI=edge` — already a documented fallback in `run_bulk.sh:13` and uses
  the free `hi-IN-SwaraNeural` / `hi-IN-MadhurNeural` voices already in `EDGE_VOICES`.

**Code hardening (recommended either way):**
- [02_tts_synthesis/synthesize.py:370-372](02_tts_synthesis/synthesize.py#L370-L372)
  treats 429 the same as any other failure (returns `False`, manifest entry skipped).
  Add early-abort if first N consecutive Sarvam calls 429 with `insufficient_quota_error` —
  otherwise the bulk script burns through the corpus producing nothing.

---

## P0-2 — Filter recursively eats its own output

**Evidence:**
```
outputs/chinese/short/train_manifest.json: 409 rows, 265 unique audio_filepaths (1.54×)
outputs/hindi/short/train_manifest.json:   420 rows, 220 unique audio_filepaths (1.91×)
```

**Root cause:** [04_quality_filter/filter.py:179](04_quality_filter/filter.py#L179)
```python
for manifest_file in in_path.rglob("manifest_*.jsonl"):
```
With `--input-dir outputs/chinese/short`, the pattern matches:
- `clean/manifest_clean.jsonl` (intended)
- `augmented/manifest_augmented.jsonl` (intended)
- `augmented_real/manifest_augmented.jsonl` (stale legacy from `scripts/augment_simple.py`)
- `manifest_filtered.jsonl` (**filter's own previous output** in the same dir)

Each `filter.py` run pulls in its prior output, so duplicate rows compound across re-runs.
NeMo will train on the same `(audio_filepath, text)` twice per epoch, biasing toward
re-ingested clips.

**Fix:** replace with explicit globs:
```python
manifest_paths = [
    in_path / "clean" / "manifest_clean.jsonl",
    in_path / "augmented" / "manifest_augmented.jsonl",
]
```
or filter out `manifest_filtered.jsonl` / non-`{clean,augmented}/` parents.

---

## P0-3 — Voice rotation change silently mutates clean dataset

**Evidence:** zh/short had 100 wavs pre-run, has 166 post-run; the smoke test ran with
`TTS_BACKEND_ZH=edge` and shows `Synthesized 66 new audio files (skipped 34 resumed)`.

**Root cause:** [02_tts_synthesis/synthesize.py:401-403](02_tts_synthesis/synthesize.py#L401-L403)
builds the filename as `synth_{idx}_{voice_id}_{hash}.wav` — the **`voice_id` is part of the
path**. Resumability ([_is_valid_wav](02_tts_synthesis/synthesize.py#L432-L441)) checks for
the file at the *new* deterministic `(idx, voice)` mapping produced by
[rotate_voices](02_tts_synthesis/synthesize.py#L421-L429). The pre-existing prototype data
was generated with the old `random.sample` rotation noted in BULK_GENERATION_PLAN.md §5 —
those filenames don't line up, so 66/100 jobs are treated as fresh work.

**Consequences for bulk:**
1. `clean/` ends up with two parallel rotation schemes co-existing (some sentences covered
   by 3 voices instead of 2). Voice distribution is no longer balanced.
2. The downstream HF dataset and NeMo manifest both inherit this skew (16 sentences in
   zh/short are now over-represented).
3. Anyone re-running smoke against a real bulk output dir will compound this — every run
   may add another tranche of new WAVs.

**Fix options:**
- **Before bulk:** wipe `outputs/{chinese,hindi}/{short,long}/clean/` and re-synth from
  scratch under `rotate_voices`. Free for zh (edge-tts), needs Sarvam top-up for hi.
- **In code:** drop `voice_id` from the filename (use just `synth_{idx:06d}_{hash}.wav`)
  and store the voice in the manifest only, so resume keys on `(idx, sentence)` regardless
  of rotation. Trade-off: lose voice info from filename.
- **In smoke test:** add an assertion that "no new clean WAVs were created" when running
  in default mode against a populated tree.

---

## P0-4 — Smoke test doesn't fail-fast on broken states

The smoke test currently asserts:
- WAV counts ≥ floor (passes even with mixed-rotation pollution).
- `passthrough == 0` in aug manifest (only detects audiomentations being uninstalled).
- `train_manifest.json` rows ≥ 1 with NeMo keys (doesn't detect 1.91× duplication).
- `load_dataset` returns DatasetDict (passes on inflated counts).

**It does not assert:**
- Sarvam returned ≥ 0 newly synthesized rows when backend=sarvam.
- `set(audio_filepath)` is unique within `train_manifest.json`.
- The pre/post clean-WAV count delta matches the expected "0 new, all resumed" for a
  default-mode run with existing data.
- Voice coverage is uniform across the pool.

**Recommended smoke-test hardening:**
```python
# After step [5/7]:
for lang in ("chinese", "hindi"):
    for bucket in ("short", "long"):
        rows = [json.loads(l) for l in open(f"outputs/{lang}/{bucket}/train_manifest.json")]
        paths = [r["audio_filepath"] for r in rows]
        assert len(paths) == len(set(paths)), \
            f"FAIL {lang}/{bucket}: duplicate audio_filepath entries"
```
And in `synthesize.py`, abort the run if backend=sarvam and the first 5 consecutive jobs
return `insufficient_quota_error`.

---

## P1-5 — `pyproject.toml` augment-extras pin contradicts CLAUDE.md

[pyproject.toml:20-22](pyproject.toml#L20-L22):
```toml
augment = [
    "audiomentations>=0.36.0",
]
```
CLAUDE.md §Environment §audiomentations:
> Pin to `<0.36`. Versions ≥0.36 pull `numpy-minmax`, which has a buggy `IS_X86_64` macro
> that tries to compile AVX intrinsics on arm64 and fails.

The currently installed 0.35.0 is what produced today's non-passthrough augmented WAVs;
it got there manually, not via `pip install -e '.[augment]'`. A fresh dev or a CI install
would land 0.36+ and silently fall back to passthrough.

**Fix:** change to `"audiomentations>=0.35,<0.36"`, add `"pydub"` (needed for
`Mp3Compression`, also in CLAUDE.md). [augment.py:42](03_augmentation/augment.py#L42)
still suggests `pip install -e '.[augment]'` as the fix — update its message too.

---

## P1-6 — `augment.py` has no resume

Every call to [03_augmentation/augment.py:88-141](03_augmentation/augment.py#L88-L141)
iterates `in_path.glob("*.wav")` and re-augments everything, rewriting
`manifest_augmented.jsonl` from scratch. CLAUDE.md claims "Augmentation: same pattern —
skip if augmented WAV exists" — that's aspirational, not implemented.

For bulk this means:
- A second `run_bulk.sh` invocation (e.g. after a Sarvam top-up partway through) re-augments
  ~14,000 clips that don't need re-augmenting (~15–30 min wasted CPU + non-determinism since
  audiomentations transforms are randomized).
- Already-shipped HF data has different aug WAVs than the next re-run produces. If the HF
  push is incremental (which `upload_folder` is), this churns the remote.

**Fix:** mirror `synthesize.py`'s `--resume` pattern — skip if `aug_path.exists()` and
non-empty.

---

## P2-7 — `augmented_real/` legacy dirs pollute filter input

`outputs/{chinese,hindi}/short/augmented_real/manifest_augmented.jsonl` (20 rows each)
were created by `scripts/augment_simple.py` as a no-deps fallback. They are not produced
by `run_bulk.sh` but are pulled into filter via the `rglob` bug (P0-2). Even after fixing
P0-2, leaving them around is confusing. Either delete or move outside the bucket tree.

---

## P2-8 — Stale docs

- `BUILDPLAN.md` (April vintage): still lists Vietnamese + Singlish as primary targets,
  uses `anthropic>=0.40.0` (the project moved to `openai`), pins `audiomentations>=0.36.0`,
  references a 4-language scaffold that no longer matches. Not load-bearing on the bulk
  run, but actively misleading.
- `README.md` Quick Start uses `pip install -e ".[prototype,tts-cloud,quality,dev]"` —
  CLAUDE.md uses the lighter `".[prototype,tts-cloud]"`. The README form pulls in
  `torch`, `openai-whisper`, `transformers` etc., which aren't needed for the bulk path
  (`run_bulk.sh` runs filter with `--skip-utmos --skip-whisper`).

---

## P2-9 — Makefile recipes are missing a space

[Makefile:18-32](Makefile#L18-L32):
```makefile
generate-text:
	$(PYTHON)01_text_corpus/generate_singlish.py ...
```
Missing space between `$(PYTHON)` and the path → shell sees `.venv/bin/python01_text_corpus/...`
as a single token. CLAUDE.md notes `make is broken` because of Xcode license; even with the
license accepted these recipes wouldn't run. Either fix the spaces or remove the Makefile to
avoid confusion.

---

## P3-10 — Silent Sarvam text truncation

[synthesize.py:349-351](02_tts_synthesis/synthesize.py#L349-L351):
```python
if len(text) > 2500:
    print(f"[WARN] Sarvam input >2500 chars ({len(text)}); truncating.")
    text = text[:2500]
```
The synthesized WAV is short; the manifest text is the truncated string; but the original
corpus text is unbounded. For bulk, this could create text↔audio mismatches that look
correct in the manifest but train the model wrong. Unlikely to trip in practice
(long Hindi ≈600 chars), but a hard `raise` (or `return False` and re-prompt at corpus
stage) is safer than silently shipping a truncated transcript.

---

## P3-11 — `tests/` is empty

The project has `pytest>=8.0.0` in the `dev` extra but no test files. Not a bulk blocker
but the smoke test is currently the *only* automated check, and it has the gaps documented
in P0-4.

---

## Recommended pre-bulk checklist (in order)

1. **Fix P0-2 (filter rglob)** — one-line change in [04_quality_filter/filter.py:179](04_quality_filter/filter.py#L179).
   Smallest patch, biggest data-quality win.
2. **Decide P0-3 (voice rotation)** — either:
   - Wipe `outputs/{chinese,hindi}/{short,long}/clean/` and re-synth, OR
   - Drop `voice_id` from filename so resume is voice-agnostic.
3. **Decide P0-1 (Sarvam)** — top up OR launch with `TTS_BACKEND_HI=edge`.
4. **Add P0-4 smoke-test assertions** — duplicate detection, Sarvam-fail abort, zero-new-clean
   assertion in default mode.
5. **Fix P1-5 (pyproject pin)** — `audiomentations>=0.35,<0.36` + `pydub`.
6. **Add P1-6 (aug resume)** — symmetry with synthesize.py.
7. **Re-run smoke in `ISOLATE=1` mode** so the assertions actually exercise text gen + TTS
   from a clean slate. Default mode is too forgiving once outputs/ is populated.
8. Optional cleanup: delete `augmented_real/`, fix or remove Makefile, refresh README/BUILDPLAN
   to drop Vietnamese/Singlish-era guidance.

Only after #1–#4 should `nohup bash scripts/run_bulk.sh` be launched.

---

## Post-fix verification (2026-05-15, smoke run `logs/smoke_20260515_173651/`)

Re-ran the smoke test against the cleaned `outputs/` tree (corpus.jsonl preserved, no
clean/augmented data). End-to-end PASS:

```
[3/7] Clean WAV counts:        100 / 100 / 100 / 100 (all four buckets)
[4/7] Aug manifests:           400 rows total, 0 passthrough
[5/7] NeMo train_manifest:     158 / 200 / 200 / 200 rows, ALL UNIQUE
                               (vs. 1.54–1.91× duplicate rate before the filter fix)
[6/7] HF staging + load_dataset:  all 8 configs load with audio decoded
[7/7] Smoke test PASSED
```

One transient warning to note: the first smoke retry hit a Python bus error at 93% of
zh/short TTS (likely thread-safety in librosa/edge-tts at `WORKERS_ZH=4`). The second
attempt resumed from the 94 already-written wavs and completed cleanly. If bulk hits this
mid-run, just re-launch — the resume logic recovers. If it becomes deterministic, set
`WORKERS_ZH=2`.

### Launch checklist

1. (Done) All P0 + P1 fixes landed.
2. (Done) Polluted clean/augmented data isolated at `outputs.polluted_20260515_173520/`.
3. (Optional) Delete `outputs.polluted_20260515_173520/` once you've confirmed bulk produces
   acceptable data: `rm -rf outputs.polluted_20260515_173520`.
4. (Optional) Populate `03_augmentation/noise_bank/{ambient,rir}/` with MUSAN + RIRS_NOISES
   if you want noise/reverb augmentation; current run uses time-stretch / pitch / gain /
   MP3-codec / bandpass only (warnings, not errors).
5. Launch bulk:
   ```bash
   HF_REPO_ZH=valsea/synthetic-asr-zh \
   HF_REPO_HI=valsea/synthetic-asr-hi \
     nohup bash scripts/run_bulk.sh > bulk_run.log 2>&1 &
   tail -f bulk_run.log
   ```
   Default targets: 1,500 short + 250 long sentences/language × 2 voices = 7,000 clean
   clips + 7,000 augmented clips = ~24 hr audio. Wall-clock estimate ~2.5–3.5 hr.
