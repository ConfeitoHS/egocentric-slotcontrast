#!/bin/bash

# Training Script for Integrated Model
# Usage: ./train.sh [options]

# Default values
DATA_DIR="/path/to/saycam/data"  # UPDATE THIS
LOG_DIR="./logs"
NUM_SLOTS=7
BATCH_SIZE=16
EP_LENGTH=8
MAX_STEPS=100000
LR=0.0004
NUM_GPUS=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --num-slots)
            NUM_SLOTS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --resume)
            RESUME_CKPT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Training Script for Integrated Model"
            echo ""
            echo "Usage: ./train.sh [options]"
            echo ""
            echo "Options:"
            echo "  --data-dir PATH      Path to data directory"
            echo "  --log-dir PATH       Path to log directory (default: ./logs)"
            echo "  --num-slots N        Number of object slots (default: 7)"
            echo "  --batch-size N       Batch size per GPU (default: 16)"
            echo "  --gpus N             Number of GPUs (default: 1)"
            echo "  --resume PATH        Resume from checkpoint"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Example:"
            echo "  ./train.sh --data-dir /data/saycam --num-slots 10 --batch-size 8"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo "====================================================================="
echo "                 INTEGRATED MODEL TRAINING"
echo "====================================================================="
echo "Configuration:"
echo "  Data directory:     $DATA_DIR"
echo "  Log directory:      $LOG_DIR"
echo "  Number of slots:    $NUM_SLOTS"
echo "  Batch size per GPU: $BATCH_SIZE"
echo "  Number of GPUs:     $NUM_GPUS"
echo "  Max steps:          $MAX_STEPS"
echo "  Learning rate:      $LR"
if [ ! -z "$RESUME_CKPT" ]; then
    echo "  Resume from:        $RESUME_CKPT"
fi
echo "====================================================================="
echo ""

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    echo "Please update DATA_DIR in this script or use --data-dir option"
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"

# Build command
CMD="python train_integrated.py integrated_saycam.yml \
    --data-dir $DATA_DIR \
    --log-dir $LOG_DIR \
    --use-optimizations \
    globals.NUM_SLOTS=$NUM_SLOTS \
    globals.BATCH_SIZE_PER_GPU=$BATCH_SIZE \
    globals.NUM_GPUS=$NUM_GPUS \
    dataset.ep_length=$EP_LENGTH \
    trainer.max_steps=$MAX_STEPS \
    optimizer.lr=$LR"

# Add resume if specified
if [ ! -z "$RESUME_CKPT" ]; then
    CMD="$CMD --continue $RESUME_CKPT"
fi

# Print command
echo "Running command:"
echo "$CMD"
echo ""

# Run training
eval $CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "====================================================================="
    echo "Training completed successfully!"
    echo "====================================================================="
else
    echo ""
    echo "====================================================================="
    echo "Training failed with error code $?"
    echo "====================================================================="
    exit 1
fi
