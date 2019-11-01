push!(LOAD_PATH,"../src/")

using Documenter, pulse_input_DDM
DocMeta.setdocmeta!(pulse_input_DDM, :DocTestSetup, :(using pulse_input_DDM); recursive=true)

# Make changes here if you add to documentation
pages = Any["Home" => Any["index.md"],
        "Basics" => Any["man/getting_the_package.md", "man/choice_observation_model.md", "man/neural_observation_model.md"],
        "Other helpful info" => Any["man/vpn_is_annoying.md"],
        "Development" => Any["man/development.md"],
        "Index" =>  Any["links.md"],
        "Functions" =>  Any["functions.md"]]

makedocs(sitename="pulse input DDM",
        modules = [pulse_input_DDM],
        doctest=true,
        authors = "Brian DePasquale",
        format = Documenter.HTML(assets = ["assets/favicon.ico"]),
        pages = pages)

deploydocs(deps = Deps.pip("mkdocs", "python-markdown-math"),
           repo = "github.com/Brody-Lab/pulse_input_DDM.git",
           devbranch = "dev", devurl = "dev")
