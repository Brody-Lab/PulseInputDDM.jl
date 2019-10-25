using Test, pulse_input_DDM, LinearAlgebra

@test compute_LL(;ntrials=10, rng=1) ≈ -3.5184243553389605
@test norm(compute_gradient(;ntrials=10, rng=1)) ≈ 5.614918559409721
