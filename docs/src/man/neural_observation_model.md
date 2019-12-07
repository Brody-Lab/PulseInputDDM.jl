# Fitting the model to neural activity

Now, let's try fiting the model using neural data.


## Data conventions

Following the same conventions as ([Working interactively on scotty via a SSH tunnel](@ref)) but `rawdata` the following two fields:

- `rawdata.St`: cell array containing the spike times of each neuron on an individual trial. The cell array will be length of the number of neurons recorded on that trial. Each entry of the cell array is a column vector containing the relative timing of spikes, in seconds. Zero seconds is the start of the click stimulus.

## Load the data and fit the model interactively

Working from the notebook you started in the previous section ([Working interactively on scotty via a SSH tunnel](@ref)), we need to create three variables to point to the data we want to fit and specify which animals and sessions we want to use:

- `data_path`: a `String` indicating the directory where the `.mat` files described above are located. For example, `data_path = ENV["HOME"]*"/Projects/pulse_input_DDM.jl/data"` where `ENV["HOME"]` is using a bash environment variable and `*` conjoins two strings (like `strjoin` in MATLAB).

### Now fit the model!

You can use the function `optimize_model` to run the model.

```julia
    pz, py = optimize_model(data)
```

## Important functions

```@docs
    compute_LL(pz::Dict{}, py::Dict{}, data::Vector{Dict{Any,Any}},
      f_str::String, n::Int; state::String="state") where {T <: Any}
```

## Fitting the model on spock instead

See ([Using spock](@ref)).
