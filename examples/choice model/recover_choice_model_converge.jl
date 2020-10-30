# # Fitting a choice model
# Blah blah blah

using pulse_input_DDM

# ### Generate some data
# Blah blah blah

θ_syn = θchoice(θz=θz(σ2_i = 0.5, B = 15., λ = -0.5, σ2_a = 50., σ2_s = 1.5,
    ϕ = 0.8, τ_ϕ = 0.05),
    bias=1., θlapse=θlapse(lapse_prob=0.05, lapse_bias=0., lapse_modbeta=0.),
    θhist=θtrialhist(h_βc = 0., h_βe= 0., h_ηc =0., h_ηe= 0.))


_, data = synthetic_data(;θ=θ_syn, ntrials=10_000);

# ### Optimize stuff
# Blah blah blah

n = 53

options, x0 = create_options_and_x0
model, = optimize(data, options, n)

# ### Compute Hessian and the confidence interavls
# Blah blah blah

H = Hessian(model, n)
CI, HPSD = CIs(model, H);
