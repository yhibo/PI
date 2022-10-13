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


python training_CBAM.py --loss dice --loss_weights 0.01 0.5 0.1
python training_CBAM.py --loss dice --loss_weights 0.01 0.5 3.5
python training_CBAM.py --loss dice --loss_weights 0.01 0.5 0.5
python training_CBAM.py --loss dice --loss_weights 0.01 0.5 1.5
python training_CBAM.py --loss dice --loss_weights 0.01 0.5 2.5
python training_CBAM.py --loss dice --loss_weights 0.01 0.5 0.05
