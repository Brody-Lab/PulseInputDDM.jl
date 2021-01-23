#  Downloading the package

You need to add the pulse\_input\_DDM package from github. Startup julia by loading the julia module on scotty or spock:

```
>> module load julia/1.5.0
>> julia
```

Next add the package in julia by entering the package management mode by typing `]`.

```julia
(v1.5) pkg > add https://github.com/Brody-Lab/pulse_input_DDM/
```

If you want the `dev` branch do

```julia
(v1.5) pkg > add https://github.com/Brody-Lab/pulse_input_DDM/ #dev
```

Another way to add the package (without typing `]`) is to do the following, in the normal julia mode:

```julia
julia > using Pkg    
julia > Pkg.add(PackageSpec(url="https://github.com/Brody-Lab/pulse_input_DDM/"))
```

In either case, you will be prompted for your github username and password. This will require that you are part of the Brody-Lab github organization.

When major modifications are made to the code base, you will need to update the package. You can do this in julia's package manager (`]`) by typing `update`.
