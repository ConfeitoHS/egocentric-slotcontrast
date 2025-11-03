#!/bin/bash
singularity exec --overlay /scratch/hl6722/egocentric/tensorboard.ext3:ro /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif /bin/bash -c "source /ext3/env.sh; tensorboard --logdir=/scratch/hl6722/egocentric/S_slotcontrast"
