#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu@compute-6-10
#$ -l gpu=3
#$ -l memoria_a_usar=20G
#$ -N tt12
#
#cargar variables de entorno para encontrar 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yhibo.radlovacki/.conda/envs/pi3/lib/
module load cuda
module load miniconda
conda activate pi3


python training.py --loss dice --loss_weights 0.01 0.5 0.1 0.01
python training.py --loss dice --loss_weights 0.01 0.5 3.5
python training.py --loss dice --loss_weights 0.01 0.5 0.5
python training.py --loss dice --loss_weights 0.01 0.5 1.5
python training.py --loss dice --loss_weights 0.01 0.5 2.5
python training.py --loss dice --loss_weights 0.01 0.5 0.05
