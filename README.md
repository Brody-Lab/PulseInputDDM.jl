# Pulse Input DDM

This is a package for inferring the parameters of drift diffusion models (DDMs) from neural activity or behavioral data collected when a subject is performing a pulse-based evidence accumulation task.

##  Downloading the package

You need to add the Pulse Input DDM package from github.

```
>> module load julia/1.5.0
>> julia
```

Next add the package in julia by entering the package management mode by typing `]`.

```julia
(v1.5) pkg > add https://github.com/Brody-Lab/PulseInputDDM/
```

Another way to add the package (without typing `]`) is to do the following, in the normal Julia mode:

```julia
julia > using Pkg    
julia > Pkg.add(PackageSpec(url="https://github.com/Brody-Lab/PulseInputDDM/"))
```

When major modifications are made to the code base, you will need to update the package. You can do this in Julia's package manager (`]`) by typing `update`.
