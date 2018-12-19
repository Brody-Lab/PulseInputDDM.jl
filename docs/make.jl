push!(LOAD_PATH, "../src")

using Documenter, poisson_neural_observation

makedocs(
    sitename="My Documentation",
    doctest=false
)

