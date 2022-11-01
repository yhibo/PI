#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu@compute-6-8
#$ -l gpu=2
#$ -l memoria_a_usar=16G
#$ -N tt1_8
#
#cargar variables de entorno para encontrar cuda
module load miniconda
conda activate pi


python training_CBAM.py --epochs 500 --batch_size 8 --loss dice --loss_weights 0.01 0.2 0.1
python training_CBAM.py --epochs 500 --batch_size 8 --loss dice --loss_weights 0.01 0.2 0.1 1.5 5
python training_CBAM.py --epochs 500 --batch_size 8 --loss dice --loss_weights 0.01 0.2 0.1 1.5 1
python training_CBAM.py --epochs 500 --batch_size 8 --loss dice --loss_weights 0.01 0.2 0.1 1.5 10