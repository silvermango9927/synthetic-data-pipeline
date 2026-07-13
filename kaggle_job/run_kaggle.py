import os
import subprocess

# 1. Install necessary libraries
print("Uninstalling incompatible libraries...")
subprocess.run("pip uninstall -y torchao", shell=True)
print("Installing libraries...")
subprocess.run("pip install -q transformers accelerate bitsandbytes peft datasets soundfile librosa pandas click jiwer", shell=True)

# 2. Write multi-GPU accelerate config
print("Configuring accelerate...")
os.makedirs("/root/.cache/huggingface/accelerate", exist_ok=True)
config_text = """
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""
with open("/root/.cache/huggingface/accelerate/default_config.yaml", "w") as f:
    f.write(config_text.strip())

# 3. Embed tokens
pat_token = "GH_PAT_PLACEHOLDER"
hf_token = "HF_TOKEN_PLACEHOLDER"

if not pat_token or pat_token.startswith("GH_PAT_"):
    raise ValueError("Error: GitHub PAT token was not embedded properly!")
if not hf_token or hf_token.startswith("HF_TOKEN_"):
    raise ValueError("Error: Hugging Face Token was not embedded properly!")

os.environ["HF_TOKEN"] = hf_token

# 4. Clone private repository using PAT
print("Cloning private repository...")
subprocess.run(f"git clone -b benchmark/scaling-laws-results https://{pat_token}@github.com/silvermango9927/synthetic-data-pipeline.git", shell=True)

# 5. Clone public Hugging Face dataset (containing all WAV files)
print("Cloning Hugging Face dataset (this may take 1-2 mins)...")
subprocess.run("git clone https://huggingface.co/datasets/silvermango9927/synthetic-asr-hi synthetic-data-pipeline/outputs/hf_datasets/synthetic-asr-hi", shell=True)

# 6. Change directory and run the training sweep
os.chdir("synthetic-data-pipeline")
print("Starting training sweep...")
subprocess.run("PYTHONPATH=. accelerate launch benchmark/scaling.py --model-name openai/whisper-large-v3-turbo --use-lora --epochs 3 --batch-size 4 --lr 5e-5 --fractions '0.1,0.2,0.4,0.6,0.8,1.0' --lang hi", shell=True)
