#  Getting the pulse input DDM package from GitHub

Now you need to add the pulse_input_DDM package from the github repository. Startup up a "anaconda julia" REPL the same way we did above when you installed the IJulia pacakge

```
    >> ssh scotty
    >> module load anacondapy/5.1.0
    >> julia
```

then add the package in julia by entering the `pkg` mode (by typing `]`)

```
    (v1.0) pkg > add https://github.com/PrincetonUniversity/pulse_input_DDM/
```

Another way to add the package (without typing `]`) is to do the following, in the normal julia mode:

```
    julia > using Pkg    
    julia > Pkg.add(PackageSpec(url="https://github.com/PrincetonUniversity/pulse_input_DDM/"))
```

In either case, you will be prompted for your github username and password. This will require that you are part of the Princeton University github organization and the Brody Lab team. If you are not, fill out [this form](https://forms.rc.princeton.edu/github) to get added and make sure your mention that you want to be added to the Brody Lab team.