#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu@compute-6-8
#$ -l gpu=1
#$ -l memoria_a_usar=20G
#$ -N tt1_8
#
#cargar variables de entorno para encontrar cuda
module load miniconda
conda activate pi


echo DeviceID: $SGE_GPU

#ejecutar binario con sus respectivos argumentos
if [ $SGE_GPU == 1 ]; then
	python trainone.py
fi