# Fitting the model to neural activity

Now, let's try fiting the model using neural data.


## Data conventions

Following the same conventions as ([Working interactively on scotty via a SSH tunnel](@ref)) but `rawdata` the following two fields:

- `rawdata.St`: cell array containing the spike times of each neuron on an individual trial. The cell array will be length of the number of neurons recorded on that trial. Each entry of the cell array is a column vector containing the relative timing of spikes, in seconds. 0 seconds is the start of the click stimulus and spikes are retained up to the end of the trial (as defined in the field “T”).
- `rawdata.cell`: cell array indicating the “cellid” (as defined by Hanks convention) number for each neuron on that trial. Will be length of the number of neurons recorded on that trial. Helpful to keep track of which neuron is which, especially when multiple sessions are stitched together (not so important in the case we are discussing, of only 1 session).

## Load the data and fit the model interactively

Working from the notebook you started in the previous section ([Working interactively on scotty via a SSH tunnel](@ref)), we need to create three variables to point to the data we want to fit and specify which animals and sessions we want to use:

- `data_path`: a `String` indicating the directory where the `.mat` files described above are located. For example, `data_path = ENV["HOME"]*"/Projects/pulse_input_DDM.jl/data"` where `ENV["HOME"]` is using a bash environment variable and `*` conjoins two strings (like `strjoin` in MATLAB).
- `ratnames`: A one-dimensional array of strings, where each entry is one that you want to use data from. For example, `ratnames = ["B068","T034"]`.
- `sessids`: A one-dimensional array of one-dimensional arrays of strings (get that!?) The "outer" 1D array should be the length of `ratnames` (thus each entry corresponds to each rat) and each "inner" array corresponds to the sessions you want to include for each rat. For example,

```
        sessids = vcat([[46331,46484,46630]], [[151801,152157,152272]])
```

uses sessions 46331, 46484 and 46630 from rat B068 and sessions 151801, 152157 and 152272 from rat T034.

### Now fit the model!

You can use the function `load_and_optimize()` to run the model.

```
    pz, py = load_and_optimize(data_path,sessids,ratnames)
```

Finally, we can save the results

```
    using JLD
    @save save_path*"/results.jld" pz py
```

where `save_path` is specified by you.

## Important functions

```@docs
    optimize_model(pz::Dict{}, py::Dict{}, data::Vector{Dict{Any,Any}}, f_str::String,
        n::Int; x_tol::Float64=1e-16, f_tol::Float64=1e-16, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        outer_iterations::Int=Int(2e3)) where {TT <: Any}

    compute_LL(pz::Vector{T}, py::Vector{Vector{Vector{T}}}, data::Vector{Dict{Any,Any}},
      n::Int, f_str::String) where {T <: Any}
```

## Fitting the model on spock instead

See ([Using spock](@ref)).
