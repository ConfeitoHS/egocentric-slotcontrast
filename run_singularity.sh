#!/bin/bash
singularity exec --nv --overlay /scratch/hl6722/egocentric/pytorch.ext3:rw --overlay /scratch/yy2694/data/saycam_transcript_frames/saycam_transcript_5fps.sqf:ro /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif /bin/bash
    