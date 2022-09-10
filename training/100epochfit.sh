#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu@compute-6-3
#$ -l gpu=2
#$ -l memoria_a_usar=20G
#$ -N tt100e
#
#cargar variables de entorno para encontrar cuda
module load miniconda
conda activate pi


python pyscript_FIT.py
