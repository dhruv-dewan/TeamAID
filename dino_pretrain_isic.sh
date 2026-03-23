#!/bin/bash

#SBATCH --job-name=dino_pretrain_isic
#SBATCH --account=heng-prj-aac
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/zt1/project/heng-prj/user/ddewan/AID/TeamAID/logs/%x-%j.out
#SBATCH --error=/scratch/zt1/project/heng-prj/user/ddewan/AID/TeamAID/logs/%x-%j.err


source /etc/profile
module purge
module load hpcc/zaratan
module load cuda/12.3.0

CONDA_ROOT="/home/ddewan/miniconda3"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

conda activate aid

echo "============================================================"
echo "Starting job ${SLURM_JOB_NAME}  (Job ID: ${SLURM_JOB_ID})"
echo "Running on host: $(hostname)"
echo "Cores per task: ${SLURM_CPUS_PER_TASK}"
echo "GPUs allocated : ${SLURM_STEP_GPUS}"
echo "Job started at : $(date)"
echo "Working directory: $(pwd)"
echo "============================================================"
echo
nvidia-smi
echo

cd /home/ddewan/dino/

python -m torch.distributed.run --nproc_per_node=2 main_dino.py \
    --arch resnet50 \
    --data_path /scratch/zt1/project/heng-prj/user/ddewan/AID/TeamAID/data/ISIC_DATA \
    --output_dir /scratch/zt1/project/heng-prj/user/ddewan/AID/TeamAID/data/dino_pretrained/full \
    --epochs 100 \
    --batch_size_per_gpu 64 \
    --optimizer sgd \
    --lr 0.03 \
    --weight_decay 1e-4 \
    --weight_decay_end 1e-4 \
    --global_crops_scale 0.14 1 \
    --local_crops_scale 0.05 0.14 \
    --norm_last_layer false

ECODE=$?

echo
echo "Job finished at: $(date)"
echo "Exit code: ${ECODE}"
echo "============================================================"
exit ${ECODE}
