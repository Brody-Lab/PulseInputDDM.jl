# # Fitting a choice model
# Blah blah blah

using pulse_input_DDM

# ### Generate some data
# Blah blah blah

θ_syn = θchoice(θz=θz(σ2_i = 0.5, B = 10., λ = -2., σ2_a = 50., σ2_s = 1.5,
    ϕ = 0.8, τ_ϕ = 0.05),
    bias=1., lapse=0.05)

_, data = synthetic_data(;θ=θ_syn, ntrials=10_000);

# ### Optimize stuff
# Blah blah blah

n = 53

options = choiceoptions(fit = vcat(trues(9)),
    lb = vcat([0., 8., -5., 0., 0., 0.01, 0.005], [-30, 0.]),
    ub = vcat([2., 30., 5., 100., 2.5, 1.2, 1.], [30, 1.]),
    x0 = vcat([0.1, 8., -0.1, 20., 0.5, 0.4, 0.008], [0.,0.01]))

model, = optimize(data, options, n)

# ### Compute Hessian and the confidence interavls
# Blah blah blah

H = Hessian(model, n)
CI, HPSD = CIs(model, H);
