#!/bin/bash
#SBATCH --qos=priority
#SBATCH --job-name=aggregate
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=16G
#SBATCH --output=logs/aggregate_%j.out
#SBATCH --error=logs/aggregate_%j.err

# Load Conda
eval "$(conda shell.bash hook)"  # Ensures conda is available in batch jobs
conda activate your_env  # Activate your Conda environment

# Run Python script
python -u C2022_aggregation.py # '-u' is unbuffered mode (i.e., for real-time output in the logs)
# python -u WS2022_aggregation.py

echo "Job completed."
