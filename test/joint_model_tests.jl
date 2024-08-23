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

x = PulseInputDDM.flatten(θ)
θy0 = vcat(vcat(θy.(data, f)...)...)
options0 = neural_options_noiseless(f)
x0=vcat([0., 30., 0. + eps(), 0., 0., 1. - eps(), 0.008], θy0)
θ0 = θneural_noiseless(x0, f)
model0 = noiseless_neuralDDM(θ0)
x0 = PulseInputDDM.flatten(θ0)
@unpack f = θ0
model, = optimize(model0, data, options0; iterations=2, outer_iterations=1)
x0 = vcat([0.1, 15., -0.1, 20., 0.5, 0.8, 0.008], PulseInputDDM.flatten(model.θ)[dimz+1:end])
options = neural_options(f)  
model = neuralDDM(θneural(x0, f), n, cross)
model, = optimize(model, data, options; iterations=2, outer_iterations=1)

options = neural_choice_options(f)

choice_neural_model = neural_choiceDDM(θneural_choice(vcat(x0[1:dimz], 0., 0., x0[dimz+1:end]), f), n, cross)

@test round(choice_loglikelihood(choice_neural_model, data), digits=2) ≈ -0.47

@test round(joint_loglikelihood(choice_neural_model, data), digits=2) ≈ -368.42

nparams, = PulseInputDDM.nθparams(f)

options = neural_choice_options(fit=vcat(falses(dimz), trues(2), falses.(nparams)...), lb=options.lb, ub=options.ub)

choice_neural_model, = choice_optimize(choice_neural_model, data, options; iterations=2, outer_iterations=1)

@test round(norm(PulseInputDDM.flatten(choice_neural_model.θ)), digits=2) ≈ 56.29

choice_neural_model = neural_choiceDDM(θneural_choice(vcat(x0[1:dimz], 0., 0., x0[dimz+1:end]), f), n, cross)

options = neural_choice_options(fit=vcat(trues(dimz), trues(2), trues.(nparams)...), lb=vcat(options.lb[1:7], -10., options.lb[9:end]), 
    ub=vcat(options.ub[1:7], 10., options.ub[9:end]))

choice_neural_model, = choice_optimize(choice_neural_model, data, options; iterations=2, outer_iterations=1)

@test round(norm(PulseInputDDM.flatten(choice_neural_model.θ)), digits=2) ≈ 56.29