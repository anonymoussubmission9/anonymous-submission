#!/bin/bash
#SBATCH -J grace-closure-pruned --mem=120GB --account=r_mdnakh --gpus=1
#SBATCH --mail-type=BEGIN,END 
#SBATCH --mail-user=nakhla054@gmail.com 

source /etc/profile.d/modules.sh
module load anaconda/3.2023.03
python runtotal.py Jsoup