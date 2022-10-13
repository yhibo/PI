#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu@compute-6-3
#$ -l gpu=1
#$ -l memoria_a_usar=20G
#$ -N tt3
#
#cargar variables de entorno para encontrar cuda
module load miniconda
conda activate clonepi


python training_CBAM.py --batch_size 4 --loss MSE --loss_weights 0.01 0.5 0.1 --shuffle_buffer_size 50