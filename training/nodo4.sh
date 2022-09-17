#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu@compute-6-4
#$ -l gpu=1
#$ -l memoria_a_usar=20G
#$ -N tt1_4
#
#cargar variables de entorno para encontrar cuda
module load miniconda
conda activate clonepi


python trainone.py
