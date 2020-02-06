# pulse input DDM

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://Brody-Lab.github.io/pulse_input_DDM/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://Brody-Lab.github.io/pulse_input_DDM/dev)
[![Build Status](https://travis-ci.com/Brody-Lab/pulse_input_DDM.svg?token=WcHBepPGGgEuyqydchVr&branch=master)](https://travis-ci.com/Brody-Lab/pulse_input_DDM)

This is a package for inferring the parameters of drift diffusion models (DDMs) from neural activity or behavioral data collected when a subject is performing a pulse-based evidence accumulation task.

###  Downloading the package

You need to add the pulse\_input\_DDM package from github. Startup julia by loading the julia module on scotty or spock:

```
>> module load julia/1.2.0
>> julia
```

Next add the package in julia by entering the package management mode by typing `]`.

```julia
(v1.2) pkg > add https://github.com/Brody-Lab/pulse_input_DDM/
```

Another way to add the package (without typing `]`) is to do the following, in the normal julia mode:

```julia
julia > using Pkg    
julia > Pkg.add(PackageSpec(url="https://github.com/Brody-Lab/pulse_input_DDM/"))
```

In either case, you will be prompted for your github username and password. This will require that you are part of the Brody-Lab github organization.

When major modifications are made to the code base, you will need to update the package. You can do this in julia's package manager (`]`) by typing `update`.
