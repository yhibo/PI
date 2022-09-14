#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu@compute-6-1
#$ -l gpu=1
#$ -l memoria_a_usar=20G
#$ -N tt10e
#
#cargar variables de entorno para encontrar cuda
module load miniconda
conda activate pi


python newtrainingTFRecord.py