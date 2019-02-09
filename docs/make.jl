push!(LOAD_PATH,"../src/")

using Documenter, pulse_input_DDM

makedocs(sitename="pulse input DDM", 
        modules = [pulse_input_DDM], 
        doctest=false,
        authors = "Brian DePasquale",
        assets = ["assets/favicon.ico"],
        pages = Any[
        "Home" => "index.md",
        "Basics" => Any[
          "man/choice_observation_model.md",
          "man/neural_observation_model.md",
          "man/using_spock.md",
          "man/aggregating_sessions.md"
         ],
        "Index" => "links.md",
        "Functions" => "functions.md"
         ])
 
deploydocs(deps   = Deps.pip("mkdocs", "python-markdown-math"),
           repo = "github.com/PrincetonUniversity/pulse_input_DDM.git",
           versions = ["stable" => "v^", "v#.#"]
          )