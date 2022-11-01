#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu@compute-6-3
#$ -l gpu=2
#$ -l memoria_a_usar=16G
#$ -N tt3
#
#cargar variables de entorno para encontrar cuda
module load miniconda
conda activate clonepi


python training_CBAM.py --epochs 500 --batch_size 6 --loss dice --loss_weights 0.1 0.2 0.2 1 5