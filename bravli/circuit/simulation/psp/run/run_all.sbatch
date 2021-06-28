#!/bin/bash

CIRCUITCONFIG=$1
PATHWAYSDIR=$2
TARGETMAP=$3
OUTPUT=$4

echo "For pathways in "$PATHWAYSDIR
for filename in $PATHWAYSDIR/*.yaml; do
    echo "run for file "$(basename $filename)
    jobname=$(basename $filename)
    sbatch -J $jobname run_one.sbatch $CIRCUITCONFIG $filename $TARGETMAP $OUTPUT
done
