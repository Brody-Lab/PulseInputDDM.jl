# ### Fitting a choice model
# Blah blah blah

using pulse_input_DDM

# ### Geneerate some data
# Blah blah blah

f, ncells, ntrials, nparams, n = "Sigmoid", [2,3], [100,200], 4, 53

θ_syn = θneural(θz = θz(σ2_i = 0.5, B = 15., λ = -0.5, σ2_a = 10., σ2_s = 1.2,
    ϕ = 0.6, τ_ϕ =  0.02),
    θy=[[Sigmoid() for n in 1:N] for N in ncells], ncells=ncells,
    nparams=nparams, f=f)

data = synthetic_data(θ_syn, ntrials)

# ### Optimize stuff
# Blah blah blah

θy0 = vcat(vcat(initialize_θy.(data, f)...)...)

options = neuraloptions(ncells=ncells,
    fit=vcat(falses(dimz), trues(sum(ncells)*nparams)),
    x0=vcat([0.1, 15., -0.1, 20., 0.5, 0.8, 0.008], θy0))

model = optimize(data, options)

x = pulse_input_DDM.flatten(model.θ)
options = neuraloptions(ncells=ncells, x0=x)

model = optimize(data, options, n)

# ### Compute Hessian
# Blah blah blah

H = Hessian(model, n)

# ### Get the CIs from the Hessian
# Blah blah blah

CI, HPSD = CIs(model, H)
