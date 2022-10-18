#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu@compute-6-2
#$ -l gpu=1
#$ -l memoria_a_usar=16G
#$ -N tt2
#
#cargar variables de entorno para encontrar cuda
module load miniconda
conda activate clonepi


python training.py --batch_size 4 --loss dice --loss_weights 0.1 0.3 0.5 0 10