# pulse input DDM

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://Brody-Lab.github.io/pulse_input_DDM/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://Brody-Lab.github.io/pulse_input_DDM/dev)
[![Build Status](https://travis-ci.com/Brody-Lab/pulse_input_DDM.svg?token=WcHBepPGGgEuyqydchVr&branch=master)](https://travis-ci.com/Brody-Lab/pulse_input_DDM)

# Introduction

Julia code for inferring a latent drift diffusion model (DDM) model from data during pulsed-based evidence accumlation tasks.

# Getting started

I suggest you run this on spock which already has Julia on it. Before you can runs jobs, you have to do a little setup on spock. Follow these steps:

* Make a ssh connection to spock.
* Load the julia module: `module load julia/1.0.0`.
* Now, start julia with `julia`. 
* At the julia prompt, you now need to add this package. Julia has a 'package manager' which you can enter into by typing `]` at the julia prompt. Then, type `add https://github.com/Brody-Lab/pulse_input_DDM/` to add this package, assuming you are a member of the Brody-Lab github organization. You will need to enter you username and password to start installing the package.

When major modifications are made to the code base, you will need to update the package. You can do this in julia's package manager (`]`) by typing `update`.
