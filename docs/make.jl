push!(LOAD_PATH,"../src/")

using Documenter, pulse_input_DDM

makedocs(sitename="pulse input DDM", modules=[pulse_input_DDM], doctest=false,
        pages = Any[
        "Home" => "index.md",
        "Getting Started" => Any[
          "man/choice_observation_model.md",
          "man/neural_observation_model.md",
          "man/using_spock.md"
         ]
         ])
 
deploydocs(deps   = Deps.pip("mkdocs", "python-markdown-math"),
           repo = "github.com/PrincetonUniversity/pulse_input_DDM.git",
           versions = ["stable" => "v^", "v#.#"]
          )
