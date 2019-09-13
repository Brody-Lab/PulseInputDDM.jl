#!/bin/bash

#SESS=(157201 157357 157507 168499)
SESS=(157357 157507 168499)
#SESS=(157357)

for s in "${SESS[@]}"; do
    FILES=$HOME/Dropbox/hanks_data_cells/$s/*
    for f in $FILES; do
        b=$(basename $f .mat)
        b2=${b#*_}
        sbatch fit_model.sh $b2 $s
        sleep 1
    done
done

