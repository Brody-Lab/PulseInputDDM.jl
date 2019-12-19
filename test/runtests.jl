using Test, pulse_input_DDM, LinearAlgebra, Flatten

θ = θchoice(θz=θz(σ2_i = 0.5, B = 15., λ = -0.5, σ2_a = 50., σ2_s = 1.5,
    ϕ = 0.8, τ_ϕ = 0.05),
    bias=1., lapse=0.05)

θ, data = synthetic_data(;θ=θ, ntrials=10, rng=1)
model = choiceDDM(θ, data)

@test round(loglikelihood(model), digits=2) ≈ -3.76
@test round(θ(data), digits=2) ≈ -3.76
@test round(norm(gradient(model)), digits=2) ≈ 13.7

options = opt(fit = vcat(trues(9)),
    lb = vcat([0., 8., -5., 0., 0., 0.01, 0.005], [-30, 0.]),
    ub = vcat([2., 30., 5., 100., 2.5, 1.2, 1.], [30, 1.]),
    x0 = vcat([0.1, 15., -0.1, 20., 0.5, 0.8, 0.008], [0.,0.01]))
model, = optimize(data; options=options, iterations=5, outer_iterations=1);
@test round(norm(flatten(model.θ)), digits=2) ≈ 25.05

## Neural model
f, Ns, trials, sess = "sig", [2,3], [100,200], 2
θ, data = default_parameters_and_data(f, sess, trials, Ns);

#want to fold this into sampling
@test round(pulse_input_DDM.compute_LL(θ, data, f), digits=2) ≈ -21492.01
model = neuralDDM(θ, data)
@test round(loglikelihood(model), digits=2) ≈ -21343.94

x = pulse_input_DDM.flatten(model.θ)
@test round(loglikelihood(x, data, Ns, f), digits=2) ≈ -21343.94
@test round(norm(gradient(model)), digits=2) ≈ 292.11
