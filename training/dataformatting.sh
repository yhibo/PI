#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu@compute-6-3
#$ -l gpu=2
#$ -l memoria_a_usar=20G
#$ -N tt10e
#
#cargar variables de entorno para encontrar cuda
module load miniconda
conda activate pi


python data_formattingTFRecord.py
