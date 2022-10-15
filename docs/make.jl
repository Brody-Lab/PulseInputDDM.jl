push!(LOAD_PATH,"../src/")

using Documenter, pulse_input_DDM
using Literate
DocMeta.setdocmeta!(pulse_input_DDM, :DocTestSetup, :(using pulse_input_DDM); recursive=true)

EXAMPLE = joinpath(@__DIR__, "..", "examples", "choice model", "recover_choice_model.jl")
OUTPUT = joinpath(@__DIR__, "src/generated")
Literate.markdown(EXAMPLE, OUTPUT)

EXAMPLE = joinpath(@__DIR__, "..", "examples", "neural model", "recover_neural_model.jl")
OUTPUT = joinpath(@__DIR__, "src/generated")
Literate.markdown(EXAMPLE, OUTPUT)

EXAMPLE = joinpath(@__DIR__, "..", "examples", "choice model", "fit_choice_model.jl")
OUTPUT = joinpath(@__DIR__, "src/generated")
Literate.markdown(EXAMPLE, OUTPUT)

# Make changes here if you add to documentation
pages = Any["Home" => Any["index.md"],
        "Basics" => Any["man/getting_the_package.md",
                "man/format_for_neural_data.md",
                "generated/fit_choice_model.md",
                "generated/recover_choice_model.md",
                "generated/recover_neural_model.md"],
        "Other helpful info" => Any["man/effcient_ssh.md",
                "man/running_notebooks_on_scotty.md"],
        "Development" => Any["man/development.md"]]

makedocs(sitename="pulse input DDM",
        modules = [pulse_input_DDM],
        doctest=true,
        authors = "Brian DePasquale",
        format = Documenter.HTML(assets = ["assets/favicon.ico"]),
        pages = pages)

#makedocs(sitename="pulse input DDM")

#deploydocs(deps = Deps.pip("mkdocs", "python-markdown-math"),
#           repo = "github.com/Brody-Lab/pulse_input_DDM.git",
#           devbranch = "dev", devurl = "dev")

#deploydocs(deps = Deps.pip("mkdocs", "python-markdown-math"),
#           repo = "github.com/Brody-Lab/pulse_input_DDM.git",
#           branch = "gh-pages", devbranch = nothing,
#           devurl = "dev", versions = ["stable" => "v^", "v#.#.#", "dev" => "dev"])

#deploydocs(deps = Deps.pip("mkdocs", "python-markdown-math"),
#           repo = "github.com/Brody-Lab/pulse_input_DDM.git")

deploydocs(repo = "Brody-Lab/pulse-input-DDM")

