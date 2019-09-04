#!/usr/bin/env bash

#SBATCH -J 'FOF_3_neural_break'
#SBATCH -o ../logs/FOF_3_neural_break.out
#SBATCH -p Brody
#SBATCH --time=48:00:00
#SBATCH --mem=64000
#SBATCH -c 44

module load anacondapy/5.1.0
source activate julia
path=$HOME/Projects/pulse_input_DDM.jl/

julia $path/scripts/by_rat.jl "FOF" 3 "neural_break"