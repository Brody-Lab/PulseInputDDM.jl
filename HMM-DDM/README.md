# pulse input DDM

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://Brody-Lab.github.io/pulse_input_DDM/stable)

This is a package for inferring the parameters of drift diffusion models (DDMs) from neural activity or behavioral data collected when a subject is performing a pulse-based evidence accumulation task.

###  Downloading the package

You need to add the pulse\_input\_DDM package from github.

```
>> module load julia/1.8.0
>> julia
```

Next add the package in julia by entering the package management mode by typing `]`.

```julia
(v1.8) pkg > add https://github.com/Brody-Lab/pulse_input_DDM/
```

Another way to add the package (without typing `]`) is to do the following, in the normal julia mode:

```julia
julia > using Pkg    
julia > Pkg.add(PackageSpec(url="https://github.com/Brody-Lab/pulse_input_DDM/"))
```

When major modifications are made to the code base, you will need to update the package. You can do this in julia's package manager (`]`) by typing `update`.
