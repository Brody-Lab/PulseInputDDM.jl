#  Getting started

You need to add the pulse_input_DDM package from github. Startup julia by loading the julia module on scotty or spock:

```
    >> module load julia/1.0.0
    >> julia
```

Next add the package in julia by entering the packagae management mode by typing `]`.

```julia
    (v1.0) pkg > add https://github.com/Brody-Lab/pulse_input_DDM/
```

Another way to add the package (without typing `]`) is to do the following, in the normal julia mode:

```julia
    julia > using Pkg    
    julia > Pkg.add(PackageSpec(url="https://github.com/Brody-Lab/pulse_input_DDM/"))
```

In either case, you will be prompted for your github username and password. This will require that you are part of the Brody-Lab github organization.