#!/bin/bash
#SBATCH --job-name=saycam_sc_s_v2          # Job name
#SBATCH --output=logs/saycam_s_%j.out   # Standard output log
#SBATCH --error=logs/saycam_s_%j.err    # Standard error log
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --mem=32GB                   # server memory requested (per node)
#SBATCH -t 20:00:00                  # Time limit (hh:mm:ss)
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --cpus-per-task=2
singularity exec --nv \
--overlay /scratch/hl6722/egocentric/pytorch.ext3:rw \
--overlay /scratch/yy2694/data/saycam_transcript_frames/saycam_transcript_5fps.sqf:ro \
/scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
/bin/bash /home/hl6722/egocentric-slotcontrast/scripts/train_S_integrated_v2.sh
