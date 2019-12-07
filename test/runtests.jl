using Test, pulse_input_DDM, LinearAlgebra

@test round(compute_LL(;ntrials=10, rng=1), digits=2) ≈ -0.86
@test round(norm(compute_gradient(;ntrials=10, rng=1)), digits=2) ≈ 31.16
