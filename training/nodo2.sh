#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu@compute-6-3
#$ -l gpu=2
#$ -l memoria_a_usar=20G
#$ -N tt2
#
#cargar variables de entorno para encontrar cuda
module load miniconda
conda activate clonepi


python training_CBAM.py --batch_size 8 --loss new --loss_weights 0.01 0.5 0.1 0.5 --shuffle_buffer_size 50
python training_CBAM.py --batch_size 8 --loss new --loss_weights 0.01 0.5 0.2 0.1 --shuffle_buffer_size 50
python training_CBAM.py --batch_size 8 --loss new --loss_weights 0.01 0.5 0.3 1 --shuffle_buffer_size 50
python training_CBAM.py --batch_size 8 --loss new --loss_weights 0.01 0.5 0.4 0.01 --shuffle_buffer_size 50
python training_CBAM.py --batch_size 8 --loss new --loss_weights 0.01 0.5 0.5 5 --shuffle_buffer_size 50
python training_CBAM.py --batch_size 8 --loss new --loss_weights 0.01 0.5 3 2 --shuffle_buffer_size 50