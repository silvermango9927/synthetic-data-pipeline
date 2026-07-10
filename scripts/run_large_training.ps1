# PowerShell script to run Whisper-Large-v3-Turbo training/sweep with LoRA + Quantization (8-bit)
# Suitable for offline or low-compute environments.

# 1. To run a single training run:
# python -m benchmark.train `
#   --model-name openai/whisper-large-v3-turbo `
#   --use-lora `
#   --load-in-8bit `
#   --epochs 3 `
#   --batch-size 4 `
#   --lr 5e-5 `
#   --output-dir outputs/benchmark/whisper_large_turbo_lora

# 2. To run the full scaling sweep:
Write-Host "Starting Whisper Large v3 Turbo Scaling Sweeps with 8-bit Quantization and LoRA..."
python -m benchmark.scaling `
  --model-name openai/whisper-large-v3-turbo `
  --use-lora `
  --load-in-8bit `
  --epochs 3 `
  --batch-size 4 `
  --lr 5e-5 `
  --fractions "0.1,0.25,0.5,1.0"
