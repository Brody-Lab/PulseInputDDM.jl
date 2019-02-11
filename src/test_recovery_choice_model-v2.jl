@everywhere using Pkg
@everywhere Pkg.activate("/mnt/bucket/people/briandd/Projects/pulse_input_DDM.jl")
@everywhere using pulse_input_DDM
using JLD, DataFrames

ntrials,dt = Int(5e4),1e-2

#parameters of the latent model
pz = DataFrames.DataFrame(generative = vcat(1e-6,10.,-0.5,40.,1.,0.8,0.02), 
    name = vcat("σ_i","B", "λ", "σ_a","σ_s","ϕ","τ_ϕ"),
    fit = vcat(falses(1),trues(4),trues(2)),
    initial = vcat(1.,5.,-5,100.,2.,0.2,0.08));

#initialize any parameter that is not going to be fit to its generative value
pz[:initial][pz[:fit] .== false] = pz[:generative][pz[:fit] .== false];

#parameters for the choice observation
pd = DataFrames.DataFrame(generative = 0.1, name = "bias", fit = true, initial = 0.);

#generate some simulated clicks times and trial durations
data = pulse_input_DDM.sample_clicks(ntrials,dt);

#simulate choices from the model given the generative parameters
pulse_input_DDM.sampled_dataset!(data, pz[:generative], pd[:generative][1]; num_reps=1, rng=4);

#compute the likelihood of the data given the generative parameters
compute_LL(pz[:generative],pd[:generative][1],data)

#find the ML parameters
pz[:final], pd[:final], opt_output, state = optimize_model(pz[:initial], pd[:initial][1], 
        pz[:fit], pd[:fit], data)

#compute the Hessian around the ML parameters
H = compute_Hessian(pz[:final],pd[:final][1],pz[:fit],pd[:fit],data)

pz[:CI_z_plus], pd[:CI_z_plus], pz[:CI_z_minus], pd[:CI_z_minus] = compute_CI(H, pz[:final], 
    pd[:final][1], pz[:fit], pd[:fit], data)

#identify which ML parameters have generative parameters within the CI 
pz[:within_bounds] = (pz[:CI_z_minus] .< pz[:generative]) .& (pz[:CI_z_plus] .> pz[:generative])
pd[:within_bounds] = (pd[:CI_z_minus] .< pd[:generative]) .& (pd[:CI_z_plus] .> pd[:generative])

save_path = path*"/data/results"
mkpath(save_path)
@save save_path*"/choice_recovery_fit_results.jld"
