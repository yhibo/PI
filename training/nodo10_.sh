#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu@compute-6-10
#$ -l gpu=3
#$ -l memoria_a_usar=20G
#$ -N tt12_
#
#cargar variables de entorno para encontrar cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/share/apps/miniconda3/lib
module load cuda
module load miniconda
conda activate pi3


python training_CBAM_KSAC.py --epochs 300 --batch_size 9 --loss old --loss_weights 0.01 0.5 2.5
python training_CBAM_KSAC.py --epochs 500 --batch_size 9 --loss old --loss_weights 0.01 0.5 2.5