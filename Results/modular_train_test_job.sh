#!/bin/sh
#BSUB -q gpua100
#BSUB -J test_model_run
### number of core
#BSUB -n 4
### specify that all cores should be on the same host
#BSUB -gpu "num=1:mode=exclusive_process"
### specify the memory needed
#BSUB -R "rusage[mem=3GB]"
### Number of hours needed
#BSUB -W 01:00
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u s232411@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o P4_%J.out
#BSUB -e P4_%J.err

echo "Running script..."

module load cuda/11.8
module load python3/3.10.13
source 02456_grp_67_venv/bin/activate
python3 scripts/modular_train_test.py
