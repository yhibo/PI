#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu@compute-6-4
#$ -l gpu=1
#$ -l memoria_a_usar=8G
#$ -N tt4
#
#cargar variables de entorno para encontrar cuda
module load miniconda
conda activate clonepi


python training_CBAM_KSAC.py --batch_size 4 --loss dice --loss_weights 0.1 0.2 0.1 0 8