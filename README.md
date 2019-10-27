# pulse input DDM

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://Brody-Lab.github.io/pulse_input_DDM/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://Brody-Lab.github.io/pulse_input_DDM/dev)
[![Build Status](https://travis-ci.com/Brody-Lab/pulse_input_DDM.svg?token=WcHBepPGGgEuyqydchVr&branch=master)](https://travis-ci.com/Brody-Lab/pulse_input_DDM)

# Introduction

Julia code for inferring a latent drift diffusion model (DDM) model from data during pulsed-based evidence accumlation tasks.

# Julia is annoying, and I use MATLAB. Why are you making me do this?

To solve for the maximum likelihood model parameters we need to compute gradients. Packages exist in Julia for doing this with automatic differentation. Packages for doing this in MATLAB stink. Also, generally MATALB stinks.

# OK. I get it. How do I get started?

I suggest you run this on spock which already has Julia on it.

Then you need to add this package by typing `Pkg.add(https://github.com/Brody-Lab/pulse_input_DDM/)`.

# Useful functions

Most users will want to work with these functions:

* `optimize_model(pz, pd, data)`: optimizes the model.
