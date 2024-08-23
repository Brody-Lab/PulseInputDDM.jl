n, cross = 53, false
ncells, ntrials = [1,2], [3,4]
f = [repeat(["Sigmoid"], N) for N in ncells]

θ = θneural(θz = θz(σ2_i = 0.5, B = 15., λ = -0.5, σ2_a = 10., σ2_s = 1.2,
    ϕ = 0.6, τ_ϕ =  0.02),
    θy=[[Sigmoid() for n in 1:N] for N in ncells], f=f);

data, = synthetic_data(θ, ntrials, ncells);
model_gen = neuralDDM(θ, n, cross);

spikes = map(x-> sum.(x), getfield.(vcat(data...), :spikes))

@test all(spikes .== [[5], [16], [4], [2, 2], [17, 15], [19, 16], [8, 13]])   

@test round(loglikelihood(model_gen, data), digits=2) ≈ -319.64

@test round(norm(gradient(model_gen, data)), digits=2) ≈ 32.68

x = PulseInputDDM.flatten(θ)
@test round(loglikelihood(x, model_gen, data), digits=2) ≈ -319.64

θy0 = vcat(vcat(θy.(data, f)...)...)
@test round(norm(θy0), digits=2) ≈ 38.43

options0 = neural_options_noiseless(f)
x0=vcat([0., 30., 0. + eps(), 0., 0., 1. - eps(), 0.008], θy0)

θ0 = θneural_noiseless(x0, f)
model0 = noiseless_neuralDDM(θ0)

@test round(loglikelihood(model0, data), digits=2) ≈ -1495.92

x0 = PulseInputDDM.flatten(θ0)
@unpack f = θ0

@test round(loglikelihood(x0, model0, data), digits=2) ≈ -1495.92

model, = fit(model0, data, options0; iterations=2, outer_iterations=1)
@test round(norm(PulseInputDDM.flatten(model.θ)), digits=2) ≈ 58.68

@test round(norm(gradient(model, data)), digits=2) ≈ 260.59

x0 = vcat([0.1, 15., -0.1, 20., 0.5, 0.8, 0.008], PulseInputDDM.flatten(model.θ)[dimz+1:end])
options = neural_options(f)  

model = neuralDDM(θneural(x0, f), n, cross)
model, = fit(model, data, options; iterations=2, outer_iterations=1)
@test round(norm(PulseInputDDM.flatten(model.θ)), digits=2) ≈ 55.93

H = Hessian(model, data; chunk_size=4)
@test round(norm(H), digits=2) ≈ 15.52

CI, HPSD = CIs(H)
@test round(norm(CI), digits=2) ≈ 690.08