#!/bin/bash

#SBATCH --job-name=test_supervised_stanford
#SBATCH --account=heng-prj-aac
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/zt1/project/heng-prj/user/ddewan/AID/TeamAID/logs/%x-%j.out

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
echo "GPUs allocated : ${CUDA_VISIBLE_DEVICES}"
echo "Job started at : $(date)"
echo "Working directory: $(pwd)"
echo "============================================================"
echo
nvidia-smi
echo

cd /scratch/zt1/project/heng-prj/user/ddewan/AID/TeamAID/

# DEBUG CUDA
export CUDA_VISIBLE_DEVICES=0
python -c "import torch; print(torch.cuda.device_count())"

python scripts/testDino_Stanford.py

echo
echo "Job finished at: $(date)"
echo "Exit code: ${ECODE}"
echo "============================================================"
exit ${ECODE}