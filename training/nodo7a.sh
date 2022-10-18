#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu@compute-6-7
#$ -l gpu=1
#$ -l memoria_a_usar=6G
#$ -N tt7a
#
#cargar variables de entorno para encontrar cuda
module load miniconda
conda activate clonepi


python training_CBAM_singlegpu.py --epochs 300 --batch_size 2 --loss dice --loss_weights 0.01 0.4 0.1 0.0001 10