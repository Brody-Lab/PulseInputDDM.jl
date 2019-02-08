# Fitting a model to neural activity

Some text describing this section
 
## Some important functions
 
```@docs
optimize_model(pz::Vector{TT},py::Vector{Vector{TT}},pz_fit,py_fit,data;
        dt::Float64=1e-2, n::Int=53, f_str="softplus",map_str::String="exp",
        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        x_tol::Float64=1e-16,f_tol::Float64=1e-16,g_tol::Float64=1e-12,
        iterations::Int=Int(5e3),show_trace::Bool=true, 
        λ0::Vector{Vector{Float64}}=Vector{Vector{Float64}}())
```
