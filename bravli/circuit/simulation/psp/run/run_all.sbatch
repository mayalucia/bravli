#!/bin/bash

CIRCUITCONFIG=$1
PATHWAYSDIR=$2
OUTPUT=$3

echo "For pathways in "$PATHWAYSDIR
for filename in $PATHWAYSDIR/*.yaml; do
    echo "run for file "$(basename $filename)
    sbatch -J $(basename $filename) run_one.sbatch $CIRCUITCONFIG $filename $OUTPUT
done