#!/bin/bash
#SBATCH --job-name=clip
#SBATCH --output=logs/%A.log
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
echo 	"Arguments:	$@"
echo -n	"Date:		"; date
echo 	"JobId:		$SLURM_JOBID"
echo	"Node:		$HOSTNAME"
echo	"Nodelist:	$SLURM_JOB_NODELIST"

# activate conda env
module purge >/dev/null 2>&1
source ../../imagecode_env/bin/activate
# Export env variables
export PYTHONBUFFERED=1

python3.9 -u $@ --job_id "$SLURM_JOBID"

