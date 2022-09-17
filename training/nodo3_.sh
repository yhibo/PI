#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu@compute-6-3
#$ -l gpu=1
#$ -l memoria_a_usar=20G
#$ -N tt1
#
#cargar variables de entorno para encontrar cuda
module load miniconda
conda activate clonepi


python trainone.py
