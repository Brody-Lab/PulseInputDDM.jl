# pulse input DDM

*Code for fitting latent drift diffusion models to pulsed input data and neural activity or behavioral observation data.*

```@contents
Pages = [
    "man/using_spock.md",
    "man/aggregating_sessions.md",
    "man/choice_observation_model.md",
    "man/neural_observation_model.md"]
Depth = 2
```

## Functions

```@docs
    optimize_model(pz::Vector{TT}, bias::TT, pz_fit_vec, bias_fit_vec,
        data; dt::Float64=1e-2, n=53, map_str::String="exp",
        x_tol::Float64=1e-16,f_tol::Float64=1e-16,g_tol::Float64=1e-12,
        iterations::Int=Int(5e3)) where {TT <: Any}
```

## Index

```@index
```
