#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --qos=short
#SBATCH --job-name=test
#SBATCH --output=ot_%j.out
#SBATCH --error=et_%j.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=3


module load anaconda/4.2.0_py3 
source activate barfPy3

# module load hpc/2015
# export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so

echo "_______________________________________________"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "_______________________________________________"

# mpirun -bootstrap slurm -n $SLURM_NTASKS python 02_wlm.py
srun --mpi=pmi2 -n $SLURM_NTASKS python test_pymofa.py
