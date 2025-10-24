#!/bin/bash
#SBATCH --job-name=movi_e_v2          # Job name
#SBATCH --output=logs/movi_e_%j.out   # Standard output log
#SBATCH --error=logs/movi_e_%j.err    # Standard error log
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=64000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu_jobs          # Request partition (DO NOT EDIT)
#SBATCH --ntasks-per-node=8
#SBATCH --exclude=turing-1
#SBATCH --gres=gpu:8                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption
# Print job information
echo "========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================="

# Navigate to project directory
cd /work/hslee/egocentric-slotcontrast

# Activate conda environment
source /work/hslee/miniconda3/bin/activate sc

# Verify environment
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Create log directory if it doesn't exist
mkdir -p logs

# Training configuration
CONFIG="configs/integrated_v2_movi_e.yaml"
DATA_DIR="./data"
LOG_DIR="./logs/movi_e_slotcontrast"

echo "========================================="
echo "Training Configuration:"
echo "Config: $CONFIG"
echo "Data Directory: $DATA_DIR"
echo "Log Directory: $LOG_DIR"
echo "========================================="

# Run training with original SlotContrast (NO POETRY)
srun torchrun --nproc_per_node=8 --nnodes=1 -m slotcontrast.train $CONFIG \
    --data-dir $DATA_DIR \
    --log-dir $LOG_DIR \
    2>&1 | tee logs/training_${SLURM_JOB_ID}.log

# Print completion information
echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
