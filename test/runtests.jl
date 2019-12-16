using Test, pulse_input_DDM, LinearAlgebra

θ, data = synthetic_data(;ntrials=10,rng=1)
model = choiceDDM(θ, data)

@test round(loglikelihood(model), digits=2) ≈ -3.76
@test round(norm(gradient(model)), digits=2) ≈ 13.7
