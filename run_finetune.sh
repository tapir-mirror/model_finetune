#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8        # 8 cores (8 cores per GPU)
#$ -l h_rt=1:0:0   # 24 hour runtime for fine-tuning
#$ -l h_vmem=11G    # 11 * 8 = 88G total RAM
#$ -l gpu=1         # request 1 GPU

# Load required modules
module load python/3.12.1-gcc-12.2.0
module load cuda
module load nano

# Create and activate virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    python -m venv .venv
fi
source .venv/bin/activate

# Install requirements if needed
if [ ! -f ".requirements_installed" ]; then
    pip install -r requirements.txt
    touch .requirements_installed
fi

# Set environment variables for better GPU utilization
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE="/data/scratch/$USER/.cache/huggingface"
export HF_HOME="/data/scratch/$USER/.cache/huggingface"

# Create cache directory if it doesn't exist
mkdir -p /data/scratch/$USER/.cache/huggingface

# Run the fine-tuning script
echo "Starting fine-tuning process..."
python main.py

# Deactivate virtual environment
deactivate 