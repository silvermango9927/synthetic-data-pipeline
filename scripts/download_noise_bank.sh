#!/usr/bin/env bash
#
# Populate the augmentation noise bank with the two standard ASR augmentation
# corpora so that stage 03 (data_generation/03_augmentation/augment.py) can apply
# *real* background noise and room reverberation instead of silently skipping
# them.
#
#   ambient/  <- MUSAN noise (+ optional music/babble)   openslr.org/resources/17
#   rir/      <- RIRS_NOISES room impulse responses        openslr.org/resources/28
#
# Both corpora are the de-facto standard for additive-noise + reverberation
# augmentation in ASR / speaker-recognition (Snyder et al. 2015 "MUSAN";
# Ko et al. 2017 "A study on data augmentation of reverberant speech").
# MUSAN files are already 16 kHz mono WAV; RIR files are WAV impulse responses.
#
# Usage:
#   bash scripts/download_noise_bank.sh                 # full download + extract
#   MAX_AMBIENT=200 MAX_RIR=200 bash scripts/download_noise_bank.sh   # cap file count
#   INCLUDE_MUSIC=1 INCLUDE_BABBLE=1 bash scripts/download_noise_bank.sh
#   CACHE_DIR=/data/corpora bash scripts/download_noise_bank.sh       # reuse a cache
#
# Idempotent: archives and already-populated targets are skipped on re-run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NOISE_BANK="${NOISE_BANK:-$REPO_ROOT/data_generation/03_augmentation/noise_bank}"
AMBIENT_DIR="$NOISE_BANK/ambient"
RIR_DIR="$NOISE_BANK/rir"
CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/.noise_cache}"

MUSAN_URL="https://openslr.org/resources/17/musan.tar.gz"
RIRS_URL="https://openslr.org/resources/28/rirs_noises.zip"

# Knobs (env-overridable).
INCLUDE_MUSIC="${INCLUDE_MUSIC:-0}"     # add MUSAN music/ to ambient/
INCLUDE_BABBLE="${INCLUDE_BABBLE:-0}"   # add MUSAN speech/ (babble) to ambient/
MAX_AMBIENT="${MAX_AMBIENT:-0}"         # 0 = keep all; otherwise random subset
MAX_RIR="${MAX_RIR:-0}"

mkdir -p "$AMBIENT_DIR" "$RIR_DIR" "$CACHE_DIR"

log()  { printf '\033[1;34m[noise-bank]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[noise-bank]\033[0m %s\n' "$*" >&2; }

# Pick a subset of files in a directory, deleting the rest, to cap disk usage.
cap_files() {
  local dir="$1" max="$2"
  [ "$max" -gt 0 ] || return 0
  local total
  total="$(find "$dir" -maxdepth 1 -name '*.wav' | wc -l | tr -d ' ')"
  [ "$total" -gt "$max" ] || return 0
  log "capping $dir to $max of $total files"
  # Deterministic subset: sort by name, keep first $max, delete the rest.
  find "$dir" -maxdepth 1 -name '*.wav' | sort | tail -n "+$((max + 1))" | while read -r f; do
    rm -f "$f"
  done
}

download() {
  local url="$1" dest="$2"
  if [ -f "$dest" ]; then
    log "archive already present: $dest"
    return 0
  fi
  log "downloading $url"
  if command -v wget >/dev/null 2>&1; then
    wget -c -O "$dest.part" "$url" && mv "$dest.part" "$dest"
  else
    curl -fL -C - -o "$dest.part" "$url" && mv "$dest.part" "$dest"
  fi
}

# ---------------------------------------------------------------- MUSAN -> ambient
if [ -n "$(find "$AMBIENT_DIR" -maxdepth 1 -name '*.wav' 2>/dev/null | head -1)" ]; then
  log "ambient/ already populated — skipping MUSAN (rm its *.wav to refresh)"
else
  MUSAN_TAR="$CACHE_DIR/musan.tar.gz"
  download "$MUSAN_URL" "$MUSAN_TAR"
  log "extracting MUSAN noise/ -> ambient/"
  # Extract only the categories we want; flatten into ambient/ so the
  # (non-recursive) populated-check in augment.py and audiomentations both see them.
  patterns=("musan/noise")
  [ "$INCLUDE_MUSIC" = "1" ]  && patterns+=("musan/music")
  [ "$INCLUDE_BABBLE" = "1" ] && patterns+=("musan/speech")
  tmp="$(mktemp -d "$CACHE_DIR/musan.XXXXXX")"
  for p in "${patterns[@]}"; do
    tar -xzf "$MUSAN_TAR" -C "$tmp" "$p" 2>/dev/null || warn "no $p in archive"
  done
  # MUSAN WAVs are unique-named (noise-free-sound-0000.wav etc) -> safe to flatten.
  find "$tmp" -name '*.wav' -exec mv -n {} "$AMBIENT_DIR/" \;
  rm -rf "$tmp"
  cap_files "$AMBIENT_DIR" "$MAX_AMBIENT"
  log "ambient/ now has $(find "$AMBIENT_DIR" -maxdepth 1 -name '*.wav' | wc -l | tr -d ' ') files"
fi

# ----------------------------------------------------------- RIRS_NOISES -> rir
if [ -n "$(find "$RIR_DIR" -maxdepth 1 -name '*.wav' 2>/dev/null | head -1)" ]; then
  log "rir/ already populated — skipping RIRS_NOISES (rm its *.wav to refresh)"
else
  RIRS_ZIP="$CACHE_DIR/rirs_noises.zip"
  download "$RIRS_URL" "$RIRS_ZIP"
  log "extracting real + simulated RIRs -> rir/"
  tmp="$(mktemp -d "$CACHE_DIR/rirs.XXXXXX")"
  # Real isotropic RIRs (measured rooms) + simulated small/medium/large rooms.
  unzip -q -o "$RIRS_ZIP" \
    'RIRS_NOISES/real_rirs_isotropic_noises/*.wav' \
    'RIRS_NOISES/simulated_rirs/*/*/*.wav' \
    -d "$tmp" 2>/dev/null || unzip -q -o "$RIRS_ZIP" -d "$tmp"
  # Simulated RIR filenames repeat across room sizes -> prefix with parent to avoid collisions.
  find "$tmp" -name '*.wav' | while read -r f; do
    base="$(basename "$f")"; parent="$(basename "$(dirname "$f")")"
    dest="$RIR_DIR/${parent}_${base}"
    [ -e "$dest" ] && dest="$RIR_DIR/${parent}_$(date +%s)_${base}"
    mv -n "$f" "$dest"
  done
  rm -rf "$tmp"
  cap_files "$RIR_DIR" "$MAX_RIR"
  log "rir/ now has $(find "$RIR_DIR" -maxdepth 1 -name '*.wav' | wc -l | tr -d ' ') files"
fi

log "done. Verify stage 03 now applies noise/reverb:"
log "  .venv/bin/python data_generation/03_augmentation/augment.py \\"
log "    --input-dir outputs/chinese/short/clean --output-dir /tmp/aug_check --variants 1"
log "  grep -o '\"augmentation\": \"[^\"]*\"' /tmp/aug_check/manifest_augmented.jsonl | sort | uniq -c"
