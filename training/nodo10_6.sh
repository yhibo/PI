#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu@compute-6-10
#$ -l gpu=3
#$ -l memoria_a_usar=20G
#$ -N tt12_6
#
#cargar variables de entorno para encontrar cuda
module load miniconda
conda activate pi


python training.py --loss dice --loss_weights 0.01 0.5 1 
python training.py --loss dice --loss_weights 0.01 0.5 2
python training.py --loss dice --loss_weights 0.01 0.5 3
python training.py --loss dice --loss_weights 0.01 0.5 4
python training.py --loss dice --loss_weights 0.01 0.1 0.1
python training.py --loss dice --loss_weights 0.01 0.1 0.2
python training.py --loss dice --loss_weights 0.01 0.1 0.3
python training.py --loss dice --loss_weights 0.01 0.1 0.4
python training.py --loss dice --loss_weights 0.01 0.1 0.5
