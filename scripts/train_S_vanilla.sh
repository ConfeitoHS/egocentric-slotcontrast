source /ext3/env.sh

echo "========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================="

cd /home/hl6722/egocentric-slotcontrast

# Verify environment
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Create log directory if it doesn't exist
mkdir -p logs

# Training configuration
CONFIG="configs/vanilla_v2_S.yaml"
DATA_DIR="./data"
LOG_DIR="/scratch/hl6722/egocentric/S_slotcontrast"
echo "========================================="
echo "Training Configuration:"
echo "Config: $CONFIG"
echo "Data Directory: $DATA_DIR"
echo "Log Directory: $LOG_DIR"
echo "========================================="

export OMP_NUM_THREADS=4
# Run training with original SlotContrast (NO POETRY)
torchrun --nproc-per-node=2 -m slotcontrast.train $CONFIG \
    --data-dir $DATA_DIR \
    --log-dir $LOG_DIR \
    2>&1 | tee logs/training_${SLURM_JOB_ID}.log

# Print completion information
echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
