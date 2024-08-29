ncells, ntrials = [1,2], [3,4]
f = [repeat(["Sigmoid"], N) for N in ncells]

θ = θneural(θz = θz(σ2_i = 0.5, B = 15., λ = -0.5, σ2_a = 10., σ2_s = 1.2,
    ϕ = 0.6, τ_ϕ =  0.02),
    θy=[[Sigmoid() for n in 1:N] for N in ncells], f=f);

data, = synthetic_data(θ, ntrials, ncells);

spikes = map(x-> sum.(x), getfield.(vcat(data...), :spikes))

@test all(spikes .== [[5], [16], [4], [2, 2], [17, 15], [19, 16], [8, 13]])   

x = PulseInputDDM.flatten(θ)
θy0 = vcat(vcat(θy.(data, f)...)...)
fitbool, lb, ub = neural_options_noiseless(f)
x0=vcat([0., 30., 0. + eps(), 0., 0., 1. - eps(), 0.008], θy0)
θ0 = θneural(x0, f)
model0 = noiseless_neuralDDM(θ=θ0, fit=fitbool, lb=lb, ub=ub)
x0 = PulseInputDDM.flatten(θ0)
@unpack f = θ0
model, = fit(model0, data; iterations=2, outer_iterations=1)
x0 = vcat([0.1, 15., -0.1, 20., 0.5, 0.8, 0.008], PulseInputDDM.flatten(model.θ)[dimz+1:end])
fitbool, lb, ub = neural_options(f)  
model = neuralDDM(θ=θneural(x0, f), fit=fitbool, lb=lb, ub=ub)
model, = fit(model, data; iterations=2, outer_iterations=1)

fitbool, lb, ub = neural_choice_options(f)

choice_neural_model = neural_choiceDDM(θ=θneural_choice(vcat(x0[1:dimz], 0., 0., x0[dimz+1:end]), f), fit=fitbool, lb=lb, ub=ub)

@test round(choice_loglikelihood(choice_neural_model, data), digits=2) ≈ -0.44

@test round(joint_loglikelihood(choice_neural_model, data), digits=2) ≈ -368.03 

nparams, = PulseInputDDM.nθparams(f)

choice_neural_model.fit, choice_neural_model.lb, choice_neural_model.ub =vcat(falses(dimz), trues(2), falses.(nparams)...), lb, ub
choice_neural_model, = choice_optimize(choice_neural_model, data; iterations=2, outer_iterations=1)

@test round(norm(PulseInputDDM.flatten(choice_neural_model.θ)), digits=2) ≈ 55.87

choice_neural_model.θ = θ=θneural_choice(vcat(x0[1:dimz], 0., 0., x0[dimz+1:end]), f)

choice_neural_model.fit, choice_neural_model.lb, choice_neural_model.ub = vcat(trues(dimz), trues(2), trues.(nparams)...),
    vcat(lb[1:7], -10., lb[9:end]), vcat(ub[1:7], 10., ub[9:end])

choice_neural_model, = choice_optimize(choice_neural_model, data; iterations=2, outer_iterations=1)

@test round(norm(PulseInputDDM.flatten(choice_neural_model.θ)), digits=2) ≈ 55.87