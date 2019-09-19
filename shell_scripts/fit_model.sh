#!/usr/bin/env bash

#SBATCH -J 'STR_1_by_cell'
#SBATCH -o ../logs/STR_1_by_cell.out
#SBATCH -p Brody
#SBATCH --time=48:00:00
#SBATCH --mem=64000
#SBATCH -c 44

module load anacondapy/5.1.0
source activate julia
path=$HOME/Projects/pulse_input_DDM.jl/

julia $path/scripts/by_rat.jl $1 $2 "by_cell"