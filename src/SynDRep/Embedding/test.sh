#!/usr/bin/bash
#SBATCH -J test
#SBATCH -n 1
#SBATCH --gres gpu:1
#SBATCH -t 48:00:00
#SBATCH -p gpu-mix
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=karim.shalaby@scai.fraunhofer.de


source activate /home/bio/groupshare/kshalaby/drep

module load CUDA

python ./test.py