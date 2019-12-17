using Test, pulse_input_DDM, LinearAlgebra

θ = θchoice(θz=θz(σ2_i = 0.5, B = 15., λ = -0.5, σ2_a = 50., σ2_s = 1.5,
    ϕ = 0.8, τ_ϕ = 0.05),
    bias=1., lapse=0.05)

θ, data = synthetic_data(;θ=θ, ntrials=10, rng=1)
model = choiceDDM(θ, data)

@test round(loglikelihood(model), digits=2) ≈ -3.76
@test round(norm(gradient(model)), digits=2) ≈ 13.7

pz, py, data, spikes = default_parameters_and_data("sig", 2, [100,200], [2,3])
@test round(compute_LL(pz["generative"], py["generative"], data, spikes, "sig", 53), digits=2) ≈ -21343.94
