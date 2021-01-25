#!/bin/bash
#SBATCH --partition=prod_p1
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --nodes=1

#SBATCH --cpus-per-task=40

#SBATCH --output="psp-%x-%j.out"
#SBATCH --error="psp-%x-%j.err"

BLUECONFIG=$1
FILENAME=$2
TARGETMAP=$3
OUTPUT=$4

BGLIBPY_RNG_MODE=Random123 \
psp -vv run\
    -c $BLUECONFIG\
    -o $OUTPUT\
    -t $TARGETMAP\
    -n 100\
    -r 100\
    -j 0\
    --dump-traces\
    --dump-amplitudes\
    $FILENAME