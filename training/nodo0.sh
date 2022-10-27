#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu@compute-6-0
#$ -l gpu=1
#$ -l memoria_a_usar=16G
#$ -N tt0
#
#cargar variables de entorno para encontrar cuda
module load miniconda
conda activate pi


python training_CBAM_KSAC.py --batch_size 4 --loss dice --loss_weights 0.01 0.1 0.1 0.0001 10
