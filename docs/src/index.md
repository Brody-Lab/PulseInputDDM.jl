# pulse input DDM

This is a package for inferring the parameters of drift diffusion models (DDMs) using gradient descent from neural activity or behavioral data collected when a subject is performing a pulse-based input evidence accumlation task.

## Getting Started

If you want to fit the model to some data but want to interact with Julia, I recommend using a Jupyter notebook on scotty via a SSH tunnel. This is somewhat involved to set up but worth the time. See [Preliminaries to fitting the model interactively on scotty](@ref) for the necessary steps.

To install the package, see [Getting the pulse input DDM package from GitHub](@ref).

If you want to fit a model non-interactively using spock, see [Using spock](@ref).

Other useful tips can be found [here](https://npcdocs.princeton.edu/index.php/Main_Page) (DUO authentication required).

## Fitting models

The basic functions you need to optimize the model parameters for choice data are described in [Fitting a model to choices](@ref). 
If you want to optimize the model parameters for neural data, look here [Fitting a model to neural activity](@ref). 

```@contents
Pages = [
    "man/using_spock.md",
    "man/choice_observation_model.md",
    "man/neural_observation_model.md"]
Depth = 2
```
