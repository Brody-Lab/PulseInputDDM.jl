```@meta
EditURL = "@__REPO_ROOT_URL__/"
```

### Fitting a choice model
Blah blah blah

```@example recover_choice_model
using pulse_input_DDM
```

### Geneerate some data
Blah blah blah

```@example recover_choice_model
pz, pd, data = default_parameters_and_data(ntrials=1000)
```

### Optimize stuff
Blah blah blah

```@example recover_choice_model
pz, pd, = optimize_model(pz, pd, data)
```

### Compute Hessian
Blah blah blah

```@example recover_choice_model
H = compute_Hessian(pz, pd, data; state="final")
```

### Get the CIs from the Hessian
Blah blah blah

```@example recover_choice_model
pz, pd = compute_CIs!(pz, pd, H)
```

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

