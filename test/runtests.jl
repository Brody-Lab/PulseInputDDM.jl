using Test, pulse_input_DDM, LinearAlgebra, Flatten, Parameters

n = 53

## Choice model
θ = θchoice(θz=θz(σ2_i = 0.5, B = 15., λ = -0.5, σ2_a = 50., σ2_s = 1.5,
    ϕ = 0.8, τ_ϕ = 0.05),
    bias=1., lapse=0.05)

θ, data = synthetic_data(;θ=θ, ntrials=10, rng=1)
model = choiceDDM(θ, data)

@test round(loglikelihood(model, n), digits=2) ≈ -3.76
@time loglikelihood(model, n)
@test round(θ(data), digits=2) ≈ -3.76
@test round(norm(gradient(model, n)), digits=2) ≈ 13.7

options = choiceoptions(fit = vcat(trues(9)),
    lb = vcat([0., 8., -5., 0., 0., 0.01, 0.005], [-30, 0.]),
    ub = vcat([2., 30., 5., 100., 2.5, 1.2, 1.], [30, 1.]),
    x0 = vcat([0.1, 15., -0.1, 20., 0.5, 0.8, 0.008], [0.,0.01]))

model = optimize(data, options, n; iterations=5, outer_iterations=1);
@test round(norm(Flatten.flatten(model.θ)), digits=2) ≈ 25.05

H = Hessian(model, n)
@test round(norm(H), digits=2) ≈ 40.64

CI, HPSD = CIs(model, H)
@test round(norm(CI), digits=2) ≈ 172.88

## Neural model
f, ncells, ntrials, nparams = "Sigmoid", [2,3], [100,200], 4

θ = θneural(θz = θz(σ2_i = 0.5, B = 15., λ = -0.5, σ2_a = 10., σ2_s = 1.2,
    ϕ = 0.6, τ_ϕ =  0.02),
    θy=[[Sigmoid() for n in 1:N] for N in ncells], ncells=ncells,
    nparams=4, f=f)

data = synthetic_data(θ, ntrials)
model = neuralDDM(θ, data)

@test round(loglikelihood(model, n), digits=2) ≈ -20334.09

x = pulse_input_DDM.flatten(θ)
@unpack ncells, nparams, f = θ
@test round(loglikelihood(x, data, ncells, nparams, f, n), digits=2) ≈ -20334.09

θ2 = unflatten(x, ncells, nparams, f)
@test round(norm(gradient(model, n)), digits=2) ≈ 270.6

#options = neuraloptions(ncells=ncells)
#@test round(norm(pulse_input_DDM.flatten(model.θ)), digits=2) ≈ 40.73
#@test round(norm(pulse_input_DDM.flatten(model.θ)), digits=2) ≈ 40.97

θy0 = vcat(vcat(initialize_θy.(data, f)...)...)
@test round(norm(θy0), digits=2) ≈ 34.52

#deterministic model

@test round(norm(gradient(model)), digits=2) ≈ 620.07

options0 = neuraloptions(ncells=ncells,
    fit=vcat(falses(dimz), trues(sum(ncells)*nparams)),
    x0=vcat([eps(), 30., 0. + eps(), eps(), eps(), 1. - eps(), 0.008], θy0))

θ0 = unflatten(options0.x0, ncells, nparams, f)
model0 = neuralDDM(θ0, data)

@test round(loglikelihood(model0), digits=2) ≈ -20578.17
x = pulse_input_DDM.flatten(θ0)
@unpack ncells, nparams, f = θ0
@test round(loglikelihood(x, data, ncells, nparams, f), digits=2) ≈ -20578.17

model = optimize(data, options0; iterations=5, outer_iterations=1)
@test round(norm(pulse_input_DDM.flatten(model.θ)), digits=2) ≈ 45.38 #new init

#optimize full model
options = neuraloptions(ncells=ncells,
    fit=vcat(falses(dimz), trues(sum(ncells)*nparams)),
    x0=vcat([0.1, 15., -0.1, 20., 0.5, 0.8, 0.008], θy0))

model = optimize(data, options, n; iterations=5, outer_iterations=1)
@test round(norm(pulse_input_DDM.flatten(model.θ)), digits=2) ≈ 42.38 #new init

H = Hessian(model, n)
@test round(norm(H), digits=2) ≈ 8348.97

CI, HPSD = CIs(model, H)
@test round(norm(CI), digits=2) ≈ 61.94
