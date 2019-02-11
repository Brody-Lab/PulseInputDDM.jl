# Working interactively on scotty via a SSH tunnel

You should be all set up to create a SSH tunnel to scotty (if not check out the section [Preliminaries to fitting the model interactively on scotty](@ref)). Follow the steps in the section [Now that everything's set up, how do I open a notebook _again_?](@ref) to create a new SSH tunnel and launch a new jupyter notebook server.

## Running the model using a notebook

OK, we're ready to analyze some data! Once you've created your SSH tunnel, open a new notebook and call it whatever you like and save it whereever you like. Add this to the first cell of your notebook

```
    using Distributed
    addprocs([some number])
```

where `[some number]` is the number of extra cores you want to have access to on scotty. `using Distributed` is necessary to use julia's parallel computation features. In the next cell, enter

```
    @everywhere using pulse_input_DDM
```

Now you have imported the package into the main namespace (which means you can use it's functionality) and you've placed it on all of the processors by the `@everywhere`.

OK, now move on to the section for fitting to choice data ([Fitting a model to choices](@ref)) or neural data ([Fitting a model to neural activity](@ref)). 