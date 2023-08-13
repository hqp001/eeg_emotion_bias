#!/bin/bash
#SBATCH -p short # partition (queue)
#SBATCH -N 1 # (leave at 1 unless using multi-node specific code) 
#SBATCH --ntasks-per-node=1 # number of cores
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048 # memory per core 
#SBATCH --job-name="myjob" # job name 
#SBATCH -o slurm.%N.%j.stdout.txt # STDOUT 
#SBATCH -e slurm.%N.%j.stderr.txt # STDERR 
#SBATCH --mail-user=hqp001@bucknell.edu # address to email 
#SBATCH --mail-type=ALL # mail events (NONE, BEGIN, END, FAIL, ALL)
module add python/3.10-deeplearn
srun python ./Gender_Bias_Emotion/dreamer/tsception/10-fold/main.py --emotion arousal --seed 2