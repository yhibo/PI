#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu@compute-6-10
#$ -l gpu=3
#$ -l memoria_a_usar=20G
#$ -N tt12_
#
#cargar variables de entorno para encontrar cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yhibo.radlovacki/.conda/envs/pi3/lib/
module load cuda
module load miniconda
conda activate pi3


python training_CBAM.py --epochs 500 --batch_size 12 --loss dice --loss_weights 0.01 0.5 0.1
python training_CBAM.py --epochs 500 --batch_size 12 --loss dice --loss_weights 0.01 0.5 0.1 1.5 5
python training_CBAM.py --epochs 500 --batch_size 12 --loss dice --loss_weights 0.01 0.5 0.1 1.5 1
python training_CBAM.py --epochs 500 --batch_size 12 --loss dice --loss_weights 0.01 0.5 0.1 1.5 2