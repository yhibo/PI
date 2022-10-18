#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu@compute-6-10
#$ -l gpu=3
#$ -l memoria_a_usar=20G
#$ -N tt12
#
#cargar variables de entorno para encontrar cuda
module load miniconda
conda activate pi


python training.py --loss dice --loss_weights 0.01 0.2 0.1 0.0001 10
python training.py --loss dice --loss_weights 0.01 0.3 0.5 0.0001 10
python training.py --loss dice --loss_weights 0.01 0.4 0.1 0.0001 10
python training.py --loss dice --loss_weights 0.01 0.2 0.5 0.0001 10
python training.py --loss dice --loss_weights 0.01 0.3 0.1 0.0001 10
python training.py --loss dice --loss_weights 0.01 0.4 0.5 0.0001 10

