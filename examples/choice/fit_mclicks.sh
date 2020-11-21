#!/usr/bin/env bash
#SBATCH -J 'mclicks'
#SBATCH -o '/scratch/ejdennis/ddm_runs/logs/mclicks_%a.out'
#SBATCH -e '/scratch/ejdennis/ddm_runs/logs/mclicks_%a.err'
#SBATCH -p Brody
#SBATCH --time=18:00:00
#SBATCH --mem=64000
#SBATCH -c 44

module load julia/1.2.0
julia -p 44 /scratch/ejdennis/pulse_input_DDM/examples/choice/new_fit_mclicks.jl
