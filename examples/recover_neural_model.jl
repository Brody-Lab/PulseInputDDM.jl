# # Fitting a neural model
# Blah blah blah

using pulse_input_DDM

# ### Geneerate some data
# Blah blah blah

f, ncells, ntrials, nparams = "Sigmoid", [1,2], [10,5], 4

θ_syn = θneural(θz = θz(σ2_i = 0.5, B = 15., λ = -0.5, σ2_a = 10., σ2_s = 1.2,
    ϕ = 0.6, τ_ϕ =  0.02),
    θy=[[Sigmoid() for n in 1:N] for N in ncells], ncells=ncells,
    nparams=nparams, f=f)

data = synthetic_data(θ_syn, ntrials);

# ### Optimize stuff
# Blah blah blah

n = 53

θy0 = vcat(vcat(initialize_θy.(data, f)...)...)

options0 = neuraloptions(ncells=ncells,
    fit=vcat(falses(dimz), trues(sum(ncells)*nparams)),
    x0=vcat([0., 30., 0. + eps(), 0., 0., 1. - eps(), 0.008], θy0))
model = optimize(data, options0; iterations=2, outer_iterations=1)

options = neuraloptions(ncells=ncells, x0=pulse_input_DDM.flatten(model.θ))
model = optimize(data, options, n; iterations=2, outer_iterations=1)

# ### Compute Hessian and the confidence interavls
# Blah blah blah

H = Hessian(model, n, chuck_size=4)
CI, HPSD = CIs(H);
