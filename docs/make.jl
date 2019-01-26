push!(LOAD_PATH,"../src/")

using Documenter, spike_data_latent_accum

makedocs(sitename="pulse_input_DDM.jl, modules=[spike_data_latent_accum], doctest=false,
        pages = Any[
        "Home" => "index.md",
        "Tutorials" => Any[
          "tutorial/page1.md",
          "tutorial/page2.md"
         ],
         "Section2" => Any[
           "sec2/page1.md",
           "sec2/page2.md"
         ]
         ])
 
deploydocs(deps   = Deps.pip("mkdocs", "python-markdown-math"),
           repo = "github.com/PrincetonUniversity/pulse_input_DDM.jl.git",
           versions = ["stable" => "v^", "v#.#"]
          )
