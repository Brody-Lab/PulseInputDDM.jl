#!/usr/bin/env bash

#SBATCH -J 'fit_choice_model'
#SBATCH -o '/scratch/ejdennis/ddm_runs/b052.out'
#SBATCH -e '/scratch/ejdennis/ddm_runs/b052.err'
#SBATCH -p Brody
#SBATCH --time=12:00:00
#SBATCH --mem=64000
#SBATCH -c 44

module load julia/1.2.0
julia -p 44 /scratch/ejdennis/pulse_input_DDM/examples/choice/fit_choice_model.jl
