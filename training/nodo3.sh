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
conda activate clonepi


python training.py --batch_size 5 --loss new --loss_weights 0.01 0.5 0.1 0.5 --shuffle_buffer_size 50