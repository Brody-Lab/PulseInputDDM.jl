#!/usr/bin/env bash

#SBATCH -J 'fit_choice_model'
#SBATCH -o ../logs/fit_choice_model_muscimol_8p.out
#SBATCH -p Brody
#SBATCH --time=12:00:00
#SBATCH --mem=64000
#SBATCH -c 44

module load julia/1.0.0
julia -p 44 ../scripts/fit_choice_model.jl
