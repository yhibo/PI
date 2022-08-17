#! /bin/bash
#$ -N Voxelmorph-BERNARDO
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -V
#$ -q gpu@compute-6-4.local
## Cantidad de gpu que voy a usar:
#$ -l gpu=1
## Memoria RAM que voy a usar:
#$ -l memoria_a_usar=8G
#
# Load gpu drivers and conda  -- Esto es un comentario
module load miniconda
source activate deep_learning
# Execute the script  ---- Esto es un comentario
hostname
echo "--------------------------------------------------------------------"
echo "------------------------ Iniciando Train  --------------------------"
echo "--------------------------------------------------------------------"
python train_script.py
# function train() {
#     nfilters='64'
#     batch_size='32'
#     epochs='100'
#     lr='1e-4'
#     net_model=$1
#     nkfolds=$2
#     id_fold=$3
#     nclass=$4
#     database=$5
#     echo "--------------------------------------------------------------------"
#     echo "------------------------ Iniciando Folds  --------------------------"
#     echo "--------------------------------------------------------------------"
#     echo "Command: python train_RVLV.py --database $database --net_model $net_model --nfilters $nfilters --batch_size $batch_size --epochs $epochs  --learning_rate $lr --nkfolds $nkfolds --verbose --id_fold $id_fold --loss gjaccardd" --nclass $nclass
#     echo "--------------------------------------------------------------------"
#     python train_RVLV.py --database $database --net_model $net_model --nfilters $nfilters --batch_size $batch_size --epochs $epochs --learning_rate $lr --nkfolds $nkfolds --verbose --id_fold $id_fold --loss gjaccardd --nclass $nclass
# }
#train autoencoder2 5 0 1 acdc
#train autoencoder2 5 1 1 acdc
#train autoencoder2 5 2 1 acdc
#train autoencoder2 5 3 1 acdc
#train autoencoder2 5 4 1 acdc
