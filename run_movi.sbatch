#!/bin/bash

#SBATCH --output=/scratch/wz3008/cvcl-related/slurm_outputs/%x_%j.out
#SBATCH --error=/scratch/wz3008/cvcl-related/slurm_outputs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wz3008@nyu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=slotcontrast_movi

module purge
singularity exec --nv \
    --overlay /scratch/wz3008/cvcl-related/saycam_transcript_5fps.sqf \
    /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif /bin/bash \
    -c "source /home/wz3008/.bashrc; cd /home/wz3008/slotcontrast; conda activate slotcontrast; \
    poetry run python -m slotcontrast.train /home/wz3008/slotcontrast/configs/movi_e.yaml --log_dir='/scratch/wz3008/cvcl-related/slotcontrast_logs'"
