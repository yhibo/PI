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


python training_CBAM.py --batch_size 8 --loss dice --loss_weights 0.01 0.2 0.1 0.0001 10
python training_CBAM.py --batch_size 8 --loss dice --loss_weights 0.01 0.3 0.5 0.0001 10
python training_CBAM.py --batch_size 8 --loss dice --loss_weights 0.01 0.4 0.1 0.0001 10
python training_CBAM.py --batch_size 8 --loss dice --loss_weights 0.01 0.2 0.5 0.0001 10
python training_CBAM.py --batch_size 8 --loss dice --loss_weights 0.01 0.3 0.1 0.0001 10
python training_CBAM.py --batch_size 8 --loss dice --loss_weights 0.01 0.4 0.5 0.0001 10