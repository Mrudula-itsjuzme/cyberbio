#!/bin/bash
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --time=${WALLTIME}
#SBATCH --mem=${MEM_GB}G
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --ntasks=${MPI_RANKS}

echo "Starting Slurm Job"
