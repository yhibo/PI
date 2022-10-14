#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu@compute-6-3
#$ -l gpu=1
#$ -l memoria_a_usar=20G
#$ -N tt3
#
#cargar variables de entorno para encontrar cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/share/apps/miniconda3/lib
module load cuda
module load miniconda
conda activate pi3


python training.py --batch_size 4 --loss MSE --loss_weights 0.01 0.5 0.1 --shuffle_buffer_size 50