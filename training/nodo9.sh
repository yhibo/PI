#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu@compute-6-9
#$ -l gpu=2
#$ -l memoria_a_usar=20G
#$ -N tt9
#
#cargar variables de entorno para encontrar cuda
module load miniconda
conda activate pi


python training_CBAM.py --epochs 500 --batch_size 8 --loss old --loss_weights 0.01 0.5 2.5