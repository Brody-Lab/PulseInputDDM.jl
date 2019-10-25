using Test, pulse_input_DDM, LinearAlgebra

@test compute_LL(;ntrials=10, rng=1) ≈ -3.5184243553389605
@test round(norm(compute_gradient(;ntrials=10, rng=1)), digits=5) ≈ 5.61492
