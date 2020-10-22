#!/usr/bin/env bash
#SBATCH -J 'mclicks'
#SBATCH -o '/scratch/ejdennis/ddm_runs/logs/mclicks_%a.out'
#SBATCH -e '/scratch/ejdennis/ddm_runs/logs/mclicks_%a.err'
#SBATCH -p Brody
#SBATCH --time=2:00:00
#SBATCH --mem=64000
#SBATCH -c 11

module load julia/1.2.0
julia -p 11 /scratch/ejdennis/pulse_input_DDM/examples/choice/fit_mclicks.jl
