# Fitting a model to choice observations

We can fit the parameters of the latent model uses animal choices.
 
## Some important functions
 
```@docs
    optimize_model(pz::Vector{TT}, bias::TT, pz_fit_vec, bias_fit_vec,
        data; dt::Float64=1e-2, n=53, map_str::String="exp",
        x_tol::Float64=1e-16,f_tol::Float64=1e-16,g_tol::Float64=1e-12,
        iterations::Int=Int(5e3))
```

