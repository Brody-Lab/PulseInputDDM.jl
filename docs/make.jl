push!(LOAD_PATH,"../src/")

using Documenter, pulse_input_DDM

#=
makedocs(sitename="pulse input DDM", 
        modules = [pulse_input_DDM], 
        doctest=false,
        authors = "Brian DePasquale",
        assets = ["assets/favicon.ico"],
        pages = Any[
        "Home" => "index.md",
        "Basics" => Any[
          "man/setting_things_up_on_scotty.md",
          "man/getting_the_package.md",
          "man/working_interactively_on_scotty.md",
          "man/choice_observation_model.md",
          "man/neural_observation_model.md",
          "man/using_spock.md"
         ],
        "Other helpful info" => Any[
           "man/vpn_is_annoying.md"
        ],
        "Development" => "man/development.md",
        "Index" => "links.md",
        "Functions" => "functions.md"
         ])

=#

makedocs(sitename="pulse input DDM", 
        modules = [pulse_input_DDM], 
        doctest=false,
        authors = "Brian DePasquale",
        assets = ["assets/favicon.ico"],
        pages = Any[
        "Home" => "index.md",
        "Basics" => Any[
          "man/getting_the_package.md",
          "man/choice_observation_model-v2.md",
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
           repo = "github.com/PrincetonUniversity/pulse_input_DDM.git",
           versions = ["stable" => "v^", "v#.#"]
          )