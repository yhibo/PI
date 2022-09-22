#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu@compute-6-10
#$ -l gpu=3
#$ -l memoria_a_usar=20G
#$ -N tt12
#
#cargar variables de entorno para encontrar cuda
module load miniconda
conda activate pi


python training.py --loss loss_MSE --loss_weights 0.01 0.5 0.9
