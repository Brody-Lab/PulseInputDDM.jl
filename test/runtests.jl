using Test, pulse_input_DDM, LinearAlgebra

<<<<<<< HEAD
@test round(compute_LL(;ntrials=10, rng=1), digits=2) ≈ -0.82
@test round(norm(compute_gradient(;ntrials=10, rng=1)), digits=2) ≈ 28.83
=======
θ = θchoice(θz=θz(σ2_i = 0.5, B = 15., λ = -0.5, σ2_a = 50., σ2_s = 1.5,
    ϕ = 0.8, τ_ϕ = 0.05),
    bias=1., lapse=0.05)

θ, data = synthetic_data(;θ=θ, ntrials=10, rng=1)
model = choiceDDM(θ, data)

@test round(loglikelihood(model), digits=2) ≈ -3.76
@test round(norm(gradient(model)), digits=2) ≈ 13.7
>>>>>>> constrained_newAPI
