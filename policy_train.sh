#!/bin/bash
#PBS -l select=1:ncpus=20:mpiprocs=20:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N policy_train
#PBS -q gpu

# Specify your email address.
#PBS -M donatien.delehelle@iit.it

module load go-1.19.4/apptainer-1.1.8
# module load go-1.19.4/singularity-1.1.8
module load nvidia/cuda-12.1.1

export START_DATE=$(date '+%D-%T')

singularity exec --nv -B /home/ddelehelle/:/home --bind /work/ddelehelle:/mnt --bind /work/ddelehelle/experiments:/exp --env-file /home/ddelehelle/myenv /work/ddelehelle/softgym.sif python -u /home/gravis/policy/train_offline.py --env_name ClothFlatten --task cloth-flatten --num_epoch 200 --step 1 --exp_name IST --learning_rate 1e-3 --out_dir /exp/policy --data_file /mnt/data/gravis/policy/block0_800000_f32_nonegrw.h5,/mnt/data/gravis/policy/block1_800000_f32_nonegrw.h5,/mnt/data/gravis/policy/block2_800000_f32_nonegrw.h5,/mnt/data/gravis/policy/block3_800000_f32_nonegrw.h5,/mnt/data/gravis/policy/block4_800000_f32_nonegrw.h5,/mnt/data/gravis/policy/block5_800000_f32_nonegrw.h5,/mnt/data/gravis/policy/block6_800000_f32_nonegrw.h5,/mnt/data/gravis/policy/block7_800000_f32_nonegrw.h5,/mnt/data/gravis/policy/block8_800000_f32_nonegrw_notfinised.h5 --run_group offline
