#!/usr/bin/env bash

#SBATCH -J 'fit_model.jl'
#SBATCH -o %j.out
#SBATCH -p Brody
#SBATCH --time=24:00:00
#SBATCH --mem=64000
#SBATCH -c 12

module load anacondapy/5.1.0
source activate julia
path=$HOME/Projects/pulse_input_DDM.jl/
#julia -p auto $path/src/test_recovery_choice_model-v2.jl $path
julia -p auto -L $path/src/test_recovery_choice_model-v2.jl $path
