#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu@compute-6-9
#$ -l gpu=2
#$ -l memoria_a_usar=20G
#$ -N tt9
#
#cargar variables de entorno para encontrar cuda
module load miniconda
conda activate pi


python training_CBAM.py --batch_size 8 --loss MSE --loss_weights 0.01 0.5 0.1 --shuffle_buffer_size 50
python training_CBAM.py --batch_size 8 --loss MSE --loss_weights 0.01 0.5 0.2 --shuffle_buffer_size 50
python training_CBAM.py --batch_size 8 --loss MSE --loss_weights 0.01 0.5 0.3 --shuffle_buffer_size 50
python training_CBAM.py --batch_size 8 --loss MSE --loss_weights 0.01 0.5 0.4 --shuffle_buffer_size 50
python training_CBAM.py --batch_size 8 --loss MSE --loss_weights 0.01 0.5 0.5 --shuffle_buffer_size 50
python training_CBAM.py --batch_size 8 --loss MSE --loss_weights 0.01 0.5 3 --shuffle_buffer_size 50