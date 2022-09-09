#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu@compute-6-10
#$ -l gpu=3
#$ -l memoria_a_usar=32G
#$ -N traintest
#
#cargar variables de entorno para encontrar cuda
module load miniconda
conda activate pi


python pyscript.py
