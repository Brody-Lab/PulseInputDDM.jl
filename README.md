# Pulse Input DDM

This is a package for inferring the parameters of drift diffusion models (DDMs) from neural activity or behavioral data collected when a subject is performing a pulse-based evidence accumulation task.

##  Downloading the package

You need to add the Pulse Input DDM package from github.

```
>> module load julia/1.5.0
>> julia
```

Next add the package in julia by entering the package management mode by typing `]`.

```julia
(v1.5) pkg > add https://github.com/Brody-Lab/PulseInputDDM/
```

Another way to add the package (without typing `]`) is to do the following, in the normal Julia mode:

```julia
julia > using Pkg    
julia > Pkg.add(PackageSpec(url="https://github.com/Brody-Lab/PulseInputDDM/"))
```

When major modifications are made to the code base, you will need to update the package. You can do this in Julia's package manager (`]`) by typing `update`.


## Fitting the model to neural activity

Now, let's try fiting the model using neural data.


### Data conventions

Following the same conventions as ([Working interactively on scotty via a SSH tunnel](@ref)) but `rawdata` should contain the following extra field:

- `rawdata.spike_times`: cell array containing the spike times of each neuron on an individual trial. The cell array will be length of the number of neurons recorded on that trial. Each entry of the cell array is a column vector containing the relative timing of spikes, in seconds. Zero seconds is the start of the click stimulus.


### Load the data and fit the model interactively

Working from the notebook you started in the previous section ([Working interactively on scotty via a SSH tunnel](@ref)), we need to create three variables to point to the data we want to fit and specify which animals and sessions we want to use:

- `data_path`: a `String` indicating the directory where the `.mat` files described above are located. For example, `data_path = ENV["HOME"]*"/Projects/pulse_input_DDM.jl/data"` where `ENV["HOME"]` is using a bash environment variable and `*` conjoins two strings (like `strjoin` in MATLAB).


### Now fit the model!

You can use the function `optimize_model` to run the model.

```julia
    pz, py = optimize_model(data)
```

### Format for neural data

First, define a path to where the data you want to fit is located.

See [Loading data and fitting a choice model](@ref) for the expected format for .MAT files if one were fitting the choice model. In addition to those fields, for a neural model rawdata should also contain an extra field:

`rawdata.spike_times`: cell array containing the spike times of each neuron on an individual trial. The cell array will be length of the number of neurons recorded on that trial. Each entry of the cell array is a column vector containing the relative timing of spikes, in seconds. Zero seconds is the start of the click stimulus. Spikes before and after the click inputs should also be included.

The convention for fitting a model with neural model is that each session should have its own .MAT file. (This constrasts with the convention for the choice model, where a single .MAT file can contain data from different session). It's just easier this way, especially if different sessions have different number of cells.

## Fitting the model to choices

### How to save your data so it can be loaded correctly

The package expects your data to live in a single .mat file which should contain a struct called `rawdata`. Each element of `rawdata` should have data for one behavioral trial and `rawdata` should contain the following fields with the specified structure:

- `rawdata.leftbups`: row-vector containing the relative timing, in seconds, of left clicks on an individual trial. 0 seconds is the start of the click stimulus.
- `rawdata.rightbups`: row-vector containing the relative timing in seconds (origin at 0 sec) of right clicks on an individual trial. 0 seconds is the start of the click stimulus.
- `rawdata.T`: the duration of the trial, in seconds. The beginning of a trial is defined as the start of the click stimulus. The end of a trial is defined based on the behavioral event “cpoke_end”. This was the Hanks convention.
- `rawdata.pokedR`: `Bool` representing the animal choice (1 = right).

### Fitting the model

Once your data is correctly formatted and you have the package added in julia, you are ready to fit the model. You need to write a slurm script to use spock's resources and a .jl file to load the data and fit the model. See examples of each below. These files are also located in the package in the `examples` directory.


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

## Contribution Guidelines

