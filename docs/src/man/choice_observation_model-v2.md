# Fitting the model on spock using choice data

OK, let's fit the model using the animal's choices!

## Data (on disk) conventions

If you're using .mat files, the package expects data to be organized in a directory in a certain way, following a certain name convention and the data in each .mat file should follow a specific convention. 

Data from session `sess` for rat `name` should be named `name_sess.mat`. All of the data you want to analyze should be located in the same directory. `name_sess.mat` should contain a single structure array called `rawdata`. Each element of `rawdata` should have data for one behavioral trials and `rawdata` should contain the following fields with the specified structure:

- `rawdata.leftbups`: row-vector containing the relative timing, in seconds, of left clicks on an individual trial. 0 seconds is the start of the click stimulus.
- `rawdata.rightbups`: row-vector containing the relative timing in seconds (origin at 0 sec) of right clicks on an individual trial. 0 seconds is the start of the click stimulus. 
- `rawdata.T`: the duration of the trial, in seconds. The beginning of a trial is defined as the start of the click stimulus. The end of a trial is defined based on the behavioral event “cpoke_end”. This was the Hanks convention.
- `rawdata.pokedR`: `Bool` representing the animal choice (1 = R).
- `rawdata.correct_dir`: `Bool` representing the correct choice (1 = R). Based on the difference in left and right clicks on that trial (not the generative gamma for that trial).

## Load the data and fit the model interactively

Working from the notebook you started in the previous section ([Working interactively on scotty via a SSH tunnel](@ref)), we need to create three variables to point to the data we want to fit and specify which animals and sessions we want to use:

- `data_path`: a `String` indicating the directory where the `.mat` files described above are located. For example, `data_path = ENV["HOME"]*"/Projects/pulse_input_DDM.jl/data"` where `ENV["HOME"]` is using a bash environment variable and `*` conjoins two strings (like `strjoin` in MATLAB).
- `ratnames`: A one-dimensional array of strings, where each entry is one that you want to use data from. For example, `ratnames = ["B068","T034"]`.
- `sessids`: A one-dimensional array of one-dimensional arrays of strings (get that!?) The "outer" 1D array should be the length of `ratnames` (thus each entry corresponds to each rat) and each "inner" array corresponds to the sessions you want to include for each rat. For example, 

```
        sessids = vcat([[46331,46484,46630]], [[151801,152157,152272]])
```

uses sessions 46331, 46484 and 46630 from rat B068 and sessions 151801, 152157 and 152272 from rat T034.

### Example slurm script

```python
s = "Python syntax highlighting"
print s
```

### Exampe .jl file

```julia
s = "Python syntax highlighting"
print s
```

### Now fit the model!

You can use the function `load_and_optimize()` to run the model.

```
    pz, pd = load_and_optimize(data_path,sessids,ratnames)
```

Finally, we can save the results

```
    using JLD
    @save save_path*"/results.jld" pz pd
```

where `save_path` is specified by you.

 


