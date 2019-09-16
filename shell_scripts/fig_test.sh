#!/usr/bin/env bash

#SBATCH -J 'fig_Test'
#SBATCH -o ../logs/fig_test.out
#SBATCH -p Brody
#SBATCH --time=2:00:00
#SBATCH --mem=64000
#SBATCH -c 2

module purge
module load anacondapy/5.1.0

source activate julia
path=$HOME/Projects/pulse_input_DDM.jl/

xvfb-run julia $path/scripts/fig_test.jl