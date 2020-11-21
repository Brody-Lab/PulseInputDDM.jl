#!/usr/bin/env bash
#SBATCH -J 'fst_%a'
#SBATCH -o '/scratch/ejdennis/ddm_runs/notfrozen/logs/fclicks_st_%a.out'
#SBATCH -e '/scratch/ejdennis/ddm_runs/notfrozen/logs/fclicks_st_%a.err'
#SBATCH -p Brody
#SBATCH --time=1:00:00
#SBATCH --mem=64000
#SBATCH -c 11

module load julia/1.2.0
julia -p 11 /scratch/ejdennis/pulse_input_DDM/examples/choice/single_trial.jl
