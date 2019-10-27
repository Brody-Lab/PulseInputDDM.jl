push!(LOAD_PATH,"../src/")

using Documenter, pulse_input_DDM
DocMeta.setdocmeta!(pulse_input_DDM, :DocTestSetup, :(using pulse_input_DDM); recursive=true)

makedocs(sitename="pulse input DDM",
        modules = [pulse_input_DDM],
        doctest=true,
        authors = "Brian DePasquale",
        format = Documenter.HTML(assets = ["assets/favicon.ico"]),
        pages = Any[
        "Home" => "index.md",
        "Basics" => Any[
          "man/getting_the_package.md",
          "man/choice_observation_model.md",
          "man/neural_observation_model.md"
         ],
        "Other helpful info" => Any[
           "man/vpn_is_annoying.md"
        ],
        "Development" => "man/development.md",
        "Index" => "links.md",
        "Functions" => "functions.md"
         ])

deploydocs(deps   = Deps.pip("mkdocs", "python-markdown-math"),
           repo = "github.com/Brody-Lab/pulse_input_DDM.git",
<<<<<<< HEAD
           devbranch = "dev", devurl = "dev")
=======
          devbranch = "dev", devurl = "dev")
>>>>>>> master
