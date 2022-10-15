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


python training.py --loss dice --loss_weights 0.1 0.1 0.1 0.0001 10
python training.py --loss dice --loss_weights 0.1 0.1 0.5 0.0001 10
python training.py --loss dice --loss_weights 0.1 0.05 0.1 0.0001 5
python training.py --loss dice --loss_weights 0.1 0.05 0.1 0.0001 10
python training.py --loss dice --loss_weights 0.1 0.05 0.1 0.0001 20
python training.py --loss dice --loss_weights 0.1 0.05 0.1 0.0001 30