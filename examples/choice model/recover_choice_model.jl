# # Fitting a choice model
# Blah blah blah

using pulse_input_DDM

# ### Generate some data
# Blah blah blah

θ_syn, data = synthetic_data(;ntrials=10);

# ### Optimize stuff
# Blah blah blah

n = 53

options, x0 = create_options_and_x0()

model, = optimize(data, options; iterations=5, outer_iterations=1)

# ### Compute Hessian and the confidence interavls
# Blah blah blah

H = Hessian(model)
CI, HPSD = CIs(H);
