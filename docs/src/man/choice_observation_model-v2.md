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
using pulse_input_DDM, MAT

println("using ", nprocs(), " cores")
data_path, save_path = "../data/", "../results/"
isdir(save_path) ? nothing : mkpath(save_path)

files = readdir(data_path)
files = files[.!isdir.(files)]
file = files[1]

println("loading data \n")
data = load_choice_data(data_path, file)

dt, use_bin_center = 1e-2, false
data = bin_clicks!(data,use_bin_center;dt=dt)

pd = Dict("name" => vcat("bias","lapse"),
          "fit" => vcat(true, true),
          "initial" => vcat(0.,0.5),
          "lb" => [-Inf, 0.],
          "ub" => [Inf, 1.])

pz = Dict("name" => ["σ_i","B", "λ", "σ_a","σ_s","ϕ","τ_ϕ"],
          "fit" => vcat(false, true, true, true, true, true, true),
          "initial" => [eps(), 10., -0.1, 20.,0.5, 1.0-eps(), 0.008],
          "lb" => [0., 8., -5., 0., 0., 0.01, 0.005],
          "ub" => [2., 40., 5., 100., 2.5, 1.2, 1.])

if isfile(save_path*file)

    println("reloading saved ML params \n")
    pz["state"] = read(matopen(save_path*file),"ML_params")[1:7]
    pd["state"] = read(matopen(save_path*file),"ML_params")[8:9]

end

println("optimize! \n")
pz, pd, converged = optimize_model_dx(pz, pd, data)
println("optimization complete \n")
println("converged: converged \n")

println("computing Hessian! \n")
pz, pd, H = compute_H_CI_dx!(pz, pd, data)

println("done. saving ML parameters! \n")
matwrite(save_path*file,
    Dict("ML_params"=> vcat(pz["final"], pd["final"]),
        "name" => vcat(pz["name"], pd["name"]),
        "CI_plus" => vcat(pz["CI_plus"], pd["CI_plus"]),
        "CI_minus" => vcat(pz["CI_minus"], pd["CI_minus"]),
        "H" => H,
        "lb"=> vcat(pz["lb"], pd["lb"]),
        "ub"=> vcat(pz["ub"], pd["ub"]),
        "fit"=> vcat(pz["fit"], pd["fit"])))
```

### Now fit the model!

You can use the function `optimize_model()` to run the model.

```
    pz, pd, = optimize_model(pz, pd, data)

```

 


