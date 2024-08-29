
ncells, ntrials = [1,2], [3,4]
f = [repeat(["Sigmoid"], N) for N in ncells]

θ = θneural(θz = θz(σ2_i = 0.5, B = 15., λ = -0.5, σ2_a = 10., σ2_s = 1.2,
    ϕ = 0.6, τ_ϕ =  0.02),
    θy=[[Sigmoid() for n in 1:N] for N in ncells], f=f);
    
fitbool,lb,ub = neural_options(f)

data, = synthetic_data(θ, ntrials, ncells);
model_gen = neuralDDM(θ=θ,fit=fitbool,lb=lb,ub=ub);

spikes = map(x-> sum.(x), getfield.(vcat(data...), :spikes))

@test all(spikes .== [[5], [16], [4], [2, 2], [17, 15], [19, 16], [8, 13]])   

@test round(loglikelihood(model_gen, data), digits=2) ≈ -319.64

@test round(norm(gradient(model_gen, data)), digits=2) ≈ 32.68

x = PulseInputDDM.flatten(θ)
@test round(loglikelihood(x, model_gen, data), digits=2) ≈ -319.64

θy0 = vcat(vcat(θy.(data, f)...)...)
@test round(norm(θy0), digits=2) ≈ 38.43

x0=vcat([0., 30., 0. + eps(), 0., 0., 1. - eps(), 0.008], θy0)

θ0 = θneural(x0, f)
fitbool,lb,ub = neural_options_noiseless(f)
model0 = noiseless_neuralDDM(θ=θ0,fit=fitbool,lb=lb,ub=ub)

@test round(loglikelihood(model0, data), digits=2) ≈ -1495.92

x0 = PulseInputDDM.flatten(θ0)
@unpack f = θ0

@test round(loglikelihood(x0, model0, data), digits=2) ≈ -1495.92

model, = fit(model0, data; iterations=2, outer_iterations=1)
@test round(norm(PulseInputDDM.flatten(model.θ)), digits=2) ≈ 58.28

#@test round(norm(gradient(model, data)), digits=2) ≈ 646.89

x0 = vcat([0.1, 15., -0.1, 20., 0.5, 0.8, 0.008], PulseInputDDM.flatten(model.θ)[dimz+1:end])
fitbool,lb,ub = neural_options(f)

model = neuralDDM(θ=θneural(x0, f),fit=fitbool,lb=lb,ub=ub)
model, = fit(model, data; iterations=2, outer_iterations=1)
@test round(norm(PulseInputDDM.flatten(model.θ)), digits=2) ≈ 55.53

H = Hessian(model, data; chunk_size=4)
@test round(norm(H), digits=2) ≈ 8.6

CI, HPSD = CIs(H)
@test round(norm(CI), digits=2) ≈ 1149.39