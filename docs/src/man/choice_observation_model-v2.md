# Fitting the model on spock using choice data

OK, let's fit the model using the animal's choices!

## Data (on disk) conventions

`name_sess.mat` should contain a single structure array called `rawdata`. Each element of `rawdata` should have data for one behavioral trials and `rawdata` should contain the following fields with the specified structure:

- `rawdata.leftbups`: row-vector containing the relative timing, in seconds, of left clicks on an individual trial. 0 seconds is the start of the click stimulus.
- `rawdata.rightbups`: row-vector containing the relative timing in seconds (origin at 0 sec) of right clicks on an individual trial. 0 seconds is the start of the click stimulus. 
- `rawdata.T`: the duration of the trial, in seconds. The beginning of a trial is defined as the start of the click stimulus. The end of a trial is defined based on the behavioral event “cpoke_end”. This was the Hanks convention.
- `rawdata.pokedR`: `Bool` representing the animal choice (1 = R).
- `rawdata.correct_dir`: `Bool` representing the correct choice (1 = R). Based on the difference in left and right clicks on that trial (not the generative gamma for that trial).

## Load the data and fit the model interactively


### Example slurm script

```python
s = "Python syntax highlighting"
print s
```

### Exampe .jl file

```julia
using pulse_input_DDM

println("using ", nprocs(), " cores")
data_path, save_path = "../data/", "../results/"
isdir(save_path) ? nothing : mkpath(save_path)

files = readdir(data_path)
files = files[.!isdir.(files)]
file = files[1]

println("loading data \n")
data = load_choice_data(data_path, file)
data = bin_clicks!(data)

pz,pd = default_parameters()

if isfile(save_path*file)

    println("reloading saved ML params \n")
    pz,pd = reload_optimization_parameters(save_path,file,pz,pd
)
end

println("optimize! \n")
pz, pd, converged = optimize_model(pz, pd, data)
println("optimization complete \n")
println("converged: converged \n")

println("computing Hessian! \n")
pz, pd, H = compute_H_CI!(pz, pd, data)

println("done. saving ML parameters! \n")
save_optimization_parameters(save_path,file,pz,pd)
```

### Now fit the model!

You can use the function `optimize_model()` to run the model.

```
    pz, pd, = optimize_model(pz, pd, data)

```

 


