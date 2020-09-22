# Developing the code

If you want to go beyond using the code to add some functionality and then share it with everyone, here's my suggested method for doing that. 

Julia's package manager makes it easy to develop packages that are hosted as git repositories. Assuming that you added this package using julia's package manager (i.e. `] add [package-name]`) it's easy to create a _second_ version of the package which you can experiment with and edit. To do this type `] dev https://github.com/Brody-Lab/pulse_input_DDM/`. This will create a second version of the package here `~/.julia/dev` which will be called from now on when you type `using pulse_input_DDM`. The directory located in `~/.julia/dev` is basically a normal git repository, but it kind of shares this special connection with julia now.

**Normally** I try to make all changes to my code base in the `dev` branch of my repository. The general idea here is that all experimental code development should take place off the `master` branch and only be merged with the master branch once its continued stability and robustness is insured. 

If you followed the above instructions, you likely only have the `master` branch in your `~/.julia/dev` directory. To 'switch over' (`checkout` in the language of git) to the `dev` branch on you local machine and to 'match it up' with the `dev` branch on github, in a shell (**not** in julia) you should navigate to `~/.julia/dev` and then type `git checkout --track origin/dev`. You can verify that you are now on the `dev` branch with `git branch` (a star should appear next to `dev`).

To jump back to the master branch, type `git checkout master`. Now all of the code if this git repo is switch over to whatever state it was in in the `master` bracnh. Now swithc over to the `dev` branch again with `git checkout dev`. Now all of the code reflects its status in the `dev` branch.

Now, the proper thing to do here is to make _another_ branch, off of the `dev` branch to make some code changes. Once those changes are complete, we can attempt to `merge` them into the `dev` branch, and once the `dev` branch is super-duper stable, we can merge _that_ with `master`. 

To make a new branch, called `bdd_dev` off of `dev` type `git checkout -b bdd_dev dev` (make sure you are on branch `dev` with `git branch`). Now from here, you can make (`git add`) and commit (`git commit`) changes as you normally would, which will all be confined to this branch. 

To push these new changes to a new remote branch in the repo, type `git push -u origin bdd_dev`. The `-u` option will 'sync' up this branch with the newly created `brians_dev` remote branch so that when you `push` and `pull` on this branch it does so from the correct remote branch.

[Here's a useful page of instructions about julia package development](https://tlienart.github.io/pub/julia/dev-pkg.html).

[Here's a useful page about git](https://www.git-tower.com/learn/git/faq/track-remote-upstream-branch).


## Adding tests

A useful way to ensure the continued stabilty of code is to include tests within a package. For julia packages, these are located in the file `tests\runtests.jl`. 

To add to the existing set of tests, add a line to `runtests.jl` something like `@testset "new_changes" begin include("new_changes_tests.jl") end` and a `.jl` file called `new_changes_tests.jl` to the `test` directory. Within `new_changes_tests.jl` include any functions that you want to test, where each function call should be preceded by an `@test`. Each `@test` should include the expected output of each function. See the examples in `runtests.jl` to get going. 

Whenever code is pushed to the repository, it is build on `travis-ci.com`, according to the specifcations on the `travis.yml` file also located in the repository. If the any of your new tests fail or if any of the existing tests fail because of changes you made, it will be immedaitely reported on the repo landing page.

# Developing the documentation

Another key step in maintaining a clear and robust code repository is documentation. I've established a few conventions that, if maintained, will allow others to use this codebase relatively easily. 

## Adding docstrings

A [docstring](https://docs.julialang.org/en/v1/manual/documentation/index.html) is a bit of code directly above a function, delineated by `"""..."""` that explains the details of how the function works. If properly formatted, they can then be access in a julia repl by typing `?`. 

## Adding doctests

A doctest can be added to a docstring to add additional explanation and to test whether a written function works as intended. See julia's [documentation](https://docs.julialang.org/en/v1/manual/documentation/index.html) page for an explanation of doctests. 

## Adding to this html documentation

I developed this documentation using [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl). This is a julia package that, once correctly instantiated, will automatically build a html webpage (this very webpage you're reading!) based on markdown files located in a specified directory in a git repo (usually `docs`) using Travis-CI whenever there is a push to that repo, and host that html webpage on the github repo on a `gh-pages` branch. 

To add a new page, you need to add a markdown files (`example.md`) in the `docs/src/man` directory. Or you can modify an existing `.md` file. Once you have you need to include it and its location in the `make.jl` file located in `docs`, so that it appears in the index of the html page.

# Merging branches

Once your branch is looking pretty good, we want to merge it with the `dev` branch, by creating a pull request on github. Several things will be done there, to ensure a robust codebase. First, any merge conflicts will have to be decided on. `travis-ci.com` will run the tests your wrote (see [Adding tests](@ref) Adding tests) above) and build the html documentation. Finally, Brian (or some other administrator) will have to approve the merge.


