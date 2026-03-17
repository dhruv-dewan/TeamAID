#!/bin/bash

#SBATCH --job-name=test_sup_ham-on-ddi
#SBATCH --account=heng-prj-aac
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=1:00:00
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

cd /scratch/zt1/project/heng-prj/user/ddewan/AID/TeamAID/

python scripts/baseline_test.py

echo
echo "Job finished at: $(date)"
echo "Exit code: ${ECODE}"
echo "============================================================"
exit ${ECODE}