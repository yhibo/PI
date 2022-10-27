#!/bin/bash

# Directorio actual es el raiz
#$ -cwd
# Nombre del proceso
#$ -N test
# stdout y stderr al mismo archivo de salida
#$ -j y
# Usar bash
#$ -S /bin/bash
# Pido la cola sumo (tiene infiniband) (Puedo usar otras colas si no requiero infiniband)
#$ -q caulle
# Pido 1GB RAM para el proceso (obligatorio)
#$ -l mem=10G
# Entorno paralelo mpi pide 10 slots (obligatorio en procesos paralelos)
#$ -pe mpi 10
# Reservo los slots a medida que otros procesos los liberan (opcional)
#$ -R y
# Tiempo de ejecución total de mi proceso (necesario si se reservan los slots)
#$ -l h_rt=1000

module load mathematica-9.0

math script.wl