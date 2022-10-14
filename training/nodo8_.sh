#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu@compute-6-8
#$ -l gpu=2
#$ -l memoria_a_usar=20G
#$ -N tt1_8
#
#cargar variables de entorno para encontrar cuda
module load miniconda
conda activate pi


python training.py --batch_size 8 --loss new --loss_weights 0.01 0.5 0.01 0 5
python training.py --batch_size 8 --loss new --loss_weights 0.01 0.5 0.001 0 5
python training.py --batch_size 8 --loss new --loss_weights 0.01 0.5 1 0 0.5
python training.py --batch_size 8 --loss new --loss_weights 0.01 0.5 2 0 0.5
python training.py --batch_size 8 --loss new --loss_weights 0.01 0.1 0.1 0 0.5
python training.py --batch_size 8 --loss new --loss_weights 0.01 0.1 0.1 0 0.1z