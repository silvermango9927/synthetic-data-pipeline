#!/bin/bash
# Download noise corpora for augmentation
set -e

NOISE_DIR="data_generation/03_augmentation/noise_bank"
mkdir -p "$NOISE_DIR/ambient" "$NOISE_DIR/rir"

echo "=== Downloading MUSAN noise corpus ==="
echo "This is ~11GB. If you want a smaller start, skip this and record your own noise."
echo "Press Ctrl+C to skip, or Enter to continue..."
read -r

wget -q --show-progress https://openslr.org/resources/17/musan.tar.gz -O /tmp/musan.tar.gz
tar -xzf /tmp/musan.tar.gz -C /tmp/
# Copy noise subset (skip music and speech to save space)
cp /tmp/musan/noise/*/*.wav "$NOISE_DIR/ambient/" 2>/dev/null || true
rm -rf /tmp/musan /tmp/musan.tar.gz

echo "=== Downloading Room Impulse Responses ==="
wget -q --show-progress https://openslr.org/resources/28/rirs_noises.zip -O /tmp/rirs.zip
unzip -q /tmp/rirs.zip -d /tmp/rirs/
find /tmp/rirs/ -name "*.wav" -exec cp {} "$NOISE_DIR/rir/" \;
rm -rf /tmp/rirs /tmp/rirs.zip

echo "Done. Noise bank at: $NOISE_DIR"
ls -la "$NOISE_DIR/ambient/" | head -5
ls -la "$NOISE_DIR/rir/" | head -5
