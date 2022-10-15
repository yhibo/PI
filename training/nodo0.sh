#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu@compute-6-0
#$ -l gpu=1
#$ -l memoria_a_usar=16G
#$ -N tt10e
#
#cargar variables de entorno para encontrar cuda
module load miniconda
conda activate clonepi


python training_CBAM.py --loss dice --loss_weights 0.01 0.1 0.1 0.0001 10
python training_CBAM.py --loss dice --loss_weights 0.01 0.1 0.5 0.0001 10
python training_CBAM.py --loss dice --loss_weights 0.01 0.05 0.1 0.0001 5
