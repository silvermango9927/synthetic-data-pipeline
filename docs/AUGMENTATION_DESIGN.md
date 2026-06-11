# Audio Augmentation Design (Stage 03)

How VALSEA's synthetic-ASR pipeline turns clean TTS audio into training data that
resembles what a real microphone in a real room actually captures — and how that
design lines up with standard practice in the ASR literature.

- **Code:** [`data_generation/03_augmentation/augment.py`](../data_generation/03_augmentation/augment.py)
- **Noise corpora:** [`scripts/download_noise_bank.sh`](../scripts/download_noise_bank.sh)
- **No-deps fallback:** [`scripts/augment_simple.py`](../scripts/augment_simple.py)

---

## 1. Why augment at all?

Our audio is synthetic (edge-tts, MiniMax, Sarvam, Qwen). TTS output is *clean*:
studio-quiet, no room, consistent level, full bandwidth, one canonical speaking
rate per voice. Real deployment audio is none of those things. An ASR model
trained only on clean TTS overfits to the TTS channel and degrades on real input.

Augmentation closes that train/serve acoustic gap by applying label-preserving
distortions: the transcript is unchanged, but the waveform is perturbed to span
the acoustic conditions the model will see in production. This is the standard
motivation for ASR data augmentation (Ko et al., 2015).

---

## 2. The pipeline

A randomized `Compose` chain. Each transform fires independently with its own
probability `p`, so every clip gets a different (possibly empty) subset. We write
`--variants` augmented copies per clean clip; with `--variants 1` the corpus
doubles (clean + 1 augmented), which is what the bulk run uses.

| # | Transform | Range | `p` | Simulates | Needs |
|---|-----------|-------|-----|-----------|-------|
| 1 | `AddBackgroundNoise` | SNR 10–25 dB | 0.7 | ambient/environmental noise | MUSAN |
| 2 | `ApplyImpulseResponse` | measured + simulated RIRs | 0.5 | room reverberation | RIRS_NOISES |
| 3 | `TimeStretch` | rate 0.9–1.1 | 0.4 | faster/slower speaking rate | — |
| 4 | `PitchShift` | ±2 semitones | 0.3 | speaker pitch variation | — |
| 5 | `Gain` | ±6 dB | 0.5 | recording-level variation | — |
| 6 | `Mp3Compression` | 32–64 kbps | 0.3 | lossy-codec / telephony artifacts | pydub+ffmpeg |
| 7 | `BandPassFilter` | 200–4000 Hz center | 0.2 | band-limited channels (phone) | — |

Transforms 1–2 are the "real-world acoustics" core. They only run when the noise
bank is populated; see §4.

**Audio invariant:** input and output are 16 kHz mono WAV (pipeline-wide
invariant, see `CLAUDE.md`). MUSAN is already 16 kHz mono; RIRs are WAV.

---

## 3. How standard is this? (literature mapping)

The recipe is a textbook offline ASR augmentation stack. Each component traces to
an established, widely-cited method:

| Component | Canonical reference | Notes on our use |
|-----------|--------------------|------------------|
| Additive background noise at controlled SNR | Snyder et al., 2015 (**MUSAN**); Hannun et al., 2014 (Deep Speech) | We use MUSAN `noise/` by default; `music/` and `speech/`(babble) are opt-in. |
| Room impulse response / reverberation | Ko et al., 2017 (*A study on data augmentation of reverberant speech*); **RIRS_NOISES** | Both measured-room and simulated RIRs. |
| Noise + RIR together as the standard pair | Snyder et al., 2018 (**Kaldi x-vector recipe**) | This MUSAN+RIRS combination is the de-facto standard augmentation in ASR/speaker-rec. |
| Speed / tempo perturbation | Ko et al., 2015 (*Audio augmentation for speech recognition*) | See the caveat in §5.1. |
| Pitch perturbation | Common in Kaldi/ESPnet augmentation configs | ±2 st is conservative and typical. |
| Gain / volume perturbation | Kaldi `volume-perturb`; standard | — |
| Codec (MP3) augmentation | Used in robustness work for telephony/streaming channels | 32–64 kbps spans low-quality channels. |
| Band-pass / band-limiting | Telephony-band robustness (≈300–3400 Hz) | Our 200–4000 Hz is a slightly wider phone band. |

**Bottom line for a methods section:** the *design* is standard and defensible —
MUSAN + RIRS + speed/pitch/gain/codec is exactly the recipe a reviewer expects.
The deviations are minor and documented below. The one thing that would *not*
survive review is shipping it with the noise bank unpopulated (§4), because then
the headline "noise augmentation" never runs.

---

## 4. The noise bank (this is the part that's easy to get wrong)

Transforms 1–2 are gated on files existing under
`data_generation/03_augmentation/noise_bank/{ambient,rir}/`. **If those dirs are
empty, the transforms are silently skipped** and "augmented" clips receive only
speed/pitch/gain/codec — no noise, no reverb. The repo ships the dirs empty (only
`.gitkeep`), because the corpora are large and gitignored.

Populate them with the standard corpora:

```bash
bash scripts/download_noise_bank.sh
# subset for a quick smoke test:
MAX_AMBIENT=200 MAX_RIR=200 bash scripts/download_noise_bank.sh
# add MUSAN music + babble to ambient/:
INCLUDE_MUSIC=1 INCLUDE_BABBLE=1 bash scripts/download_noise_bank.sh
```

| Corpus | Source | Lands in | License |
|--------|--------|----------|---------|
| MUSAN | openslr.org/resources/17 | `ambient/` | CC BY 4.0 |
| RIRS_NOISES | openslr.org/resources/28 | `rir/` | Apache 2.0 |

**Guardrail.** Run a research dataset with `--require-noise` so the job *fails
loudly* if the bank is empty, instead of quietly producing un-noised data:

```bash
.venv/bin/python data_generation/03_augmentation/augment.py \
  --input-dir outputs/chinese/short/clean \
  --output-dir outputs/chinese/short/augmented \
  --variants 1 --require-noise
```

Verify it actually fired (should show `AddBackgroundNoise` / `ApplyImpulseResponse`):

```bash
grep -o '"augmentation": "[^"]*"' outputs/<lang>/<bucket>/augmented/manifest_augmented.jsonl \
  | tr ';' '\n' | grep -oE '[A-Za-z]+\(' | sort | uniq -c
```

---

## 5. Deviations from the canonical recipe (disclose these)

### 5.1 Time-stretch ≠ speed perturbation
Ko et al. (2015) "speed perturbation" *resamples* the signal, which couples tempo,
pitch, and duration (0.9× speed → longer, lower-pitched). We use audiomentations
`TimeStretch`, which changes tempo **at constant pitch**, plus a separate
`PitchShift`. The two together cover a similar space but are not identical to the
cited method. State this rather than citing Ko (2015) as if it were a drop-in.

### 5.2 SNR range is clean-skewed
Default SNR is 10–25 dB. The Kaldi/MUSAN recipe goes lower (noise ~0–15 dB, babble
~13–20 dB, music ~5–15 dB), i.e. it includes harder, noisier examples. 10–25 dB is
appropriate if production audio is relatively clean; for robustness to noisy
field audio, lower the floor:

```bash
--min-snr-db 0 --max-snr-db 20
```

### 5.3 No SpecAugment
We do **not** apply SpecAugment (Park et al., 2019 — time/frequency masking on the
spectrogram). That is deliberate: SpecAugment is a *training-time* augmentation
applied on-the-fly in the trainer on log-mel features, not an offline waveform
transform. Confirm the downstream VALSEA trainer applies it; if it doesn't, that
is a real gap to add there, not here.

### 5.4 Offline / static, not on-the-fly
We bake a fixed number of augmented WAVs to disk rather than re-sampling
augmentation per epoch. Offline is reproducible and simple; on-the-fly gives more
diversity (a clip differs every epoch). With `--seed` fixed, our offline set is a
deterministic, auditable artifact — a reasonable trade for a *dataset* deliverable
(vs. a training-loop concern).

---

## 6. Reproducibility & provenance

Both are required for the augmentation to be defensible in a paper:

- **Seeding.** `--seed` (default 42) seeds `random` and `numpy.random`, the two
  RNGs audiomentations draws from. Same seed + same inputs ⇒ bit-identical output.
  The seed is recorded in every manifest row.
- **Per-clip provenance.** Each manifest row's `augmentation` field lists exactly
  which transforms fired on that clip and with what parameters, e.g.:

  ```
  AddBackgroundNoise(snr_db=14.1,noise_file_path=.../noise_0.wav,...);Gain(amplitude_ratio=0.57);Mp3Compression(bitrate=64)
  ```

  This replaces the old opaque `"variant_0"` label, so you can audit the exact
  distribution of applied transforms and reproduce any single clip. A clip where
  no transform fired is labeled `none`; a pass-through (audiomentations missing) is
  labeled `passthrough`, and the run prints a count of these so they can't slip
  through unnoticed.

---

## 7. Environment notes

- **audiomentations on Apple Silicon:** pin `<0.36` (≥0.36 pulls `numpy-minmax`,
  which fails to build on arm64 — not an Xcode-license issue). Install:
  `pip install 'audiomentations<0.36' pydub`. The `[augment]` extra in
  `pyproject.toml` pins `>=0.36` and is **not** usable on this arm64 box.
- **`Mp3Compression`** needs `pydub` + `ffmpeg`. If `pydub` is missing the pipeline
  now drops just that transform with a warning (it used to crash mid-loop).
- **No-deps fallback:** `scripts/augment_simple.py` mirrors transforms 3–7 using
  librosa/scipy, but its "noise" is **white Gaussian noise**, not real
  environmental noise — fine for a smoke test, not for a research dataset. Prefer
  the canonical path with a populated noise bank.

---

## 8. References

- Ko, Peddinti, Povey, Khudanpur (2015). *Audio Augmentation for Speech Recognition.* INTERSPEECH.
- Ko, Peddinti, Povey, Seltzer, Khudanpur (2017). *A Study on Data Augmentation of Reverberant Speech for Robust Speech Recognition.* ICASSP. (RIRS_NOISES)
- Snyder, Chen, Povey (2015). *MUSAN: A Music, Speech, and Noise Corpus.* arXiv:1510.08484. (OpenSLR 17)
- Snyder, Garcia-Romero, Sell, Povey, Khudanpur (2018). *X-Vectors: Robust DNN Embeddings for Speaker Recognition.* ICASSP. (MUSAN+RIRS recipe)
- Park, Chan, Zhang, Chiu, Zoph, Cubuk, Le (2019). *SpecAugment.* INTERSPEECH.
- Hannun et al. (2014). *Deep Speech.* arXiv:1412.5567. (noise synthesis for ASR)
