# Fitting the model to choices

## How to save your data so it can be loaded correctly 

The package expects your data to live in a single .mat file which should contain a struct called `rawdata`. Each element of `rawdata` should have data for one behavioral trial and `rawdata` should contain the following fields with the specified structure:

- `rawdata.leftbups`: row-vector containing the relative timing, in seconds, of left clicks on an individual trial. 0 seconds is the start of the click stimulus.
- `rawdata.rightbups`: row-vector containing the relative timing in seconds (origin at 0 sec) of right clicks on an individual trial. 0 seconds is the start of the click stimulus. 
- `rawdata.T`: the duration of the trial, in seconds. The beginning of a trial is defined as the start of the click stimulus. The end of a trial is defined based on the behavioral event “cpoke_end”. This was the Hanks convention.
- `rawdata.pokedR`: `Bool` representing the animal choice (1 = right).

## Fitting the model

Once your data is corretly formatted and you have the package added in julia, you are ready to fit the model. You need to write a slurm script to use spock's resources and a .jl file to load the data and fit the model. See examples of each below. These files are also located in the package in the `demos` directory.

### Example slurm script

This will start a job called `fit_choice_model`. Output will be written to a log file called `fit_choice_model.out`. This will run on the `Brody` partition of spock for 12 hours, using 44 cores and 64 GB of memory. You'll notice that we load the julia module (like we did when we added the package) and then we call julia (`-p 44` uses the 44 cores) and ask it to run the .jl file.

```
#!/usr/bin/env bash

#SBATCH -J 'fit_choice_model'
#SBATCH -o ../logs/fit_choice_model.out
#SBATCH -p Brody
#SBATCH --time=12:00:00
#SBATCH --mem=64000
#SBATCH -c 44

module load julia/1.0.0
julia -p 44 ../scripts/fit_choice_model.jl
```

### Example .jl file

See comments in the script to understand what each line is doing.

```julia
#use the resources of the package
using pulse_input_DDM

println("using ", nprocs(), " cores")

#define useful paths to the data and to a directory to save results
data_path, save_path = "../data/dmFC_muscimol/", "../results/dmFC_muscimol/"

#if the directory that you are saving to doesn't exist, make it
isdir(save_path) ? nothing : mkpath(save_path)

#read the name of the file located in the data directory
files = readdir(data_path)
files = files[.!isdir.(files)]
file = files[1]

#load your data
data = load_choice_data(data_path, file)

#generate default parameters for initializing the optimization
pz, pd = default_parameters()

#if you've already ran the optimization once and want to restart from where you stoped, this will reload those parameters
if isfile(save_path*file)
    pz, pd = reload_optimization_parameters(save_path, file, pz, pd)    
end

#run the optimization
pz, pd, = optimize_model(pz, pd, data)

#compute the Hessian around the ML solution, for confidence intervals
H = compute_Hessian(pz, pd, data; state="final")

#compute confidence intervals
pz, pd = compute_CIs!(pz, pd, H)

#save results
save_optimization_parameters(save_path,file,pz,pd)
```

### Getting help

To get more details about how any function in this package works, in julia you can type `?` and then the name of the function. Documentation will display in the REPL.
