#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu@compute-6-3
#$ -l gpu=1
#$ -l memoria_a_usar=12G
#$ -N tt3
#
#cargar variables de entorno para encontrar cuda
module load miniconda
conda activate clonepi


python training_CBAM_KSAC.py --batch_size 4 --loss dice --loss_weights 0.1 0.2 0.5 0.00001 10