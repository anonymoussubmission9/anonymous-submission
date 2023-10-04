#!/bin/bash
#SBATCH -J grace-closure-pruned --mem=120GB --account=r_mdnakh --gpus=40gb:1
#SBATCH --mail-type=BEGIN,END 
#SBATCH --mail-user=nakhla054@gmail.com 

source /etc/profile.d/modules.sh
module load anaconda/3.2022.10
python runtotal.py Cli