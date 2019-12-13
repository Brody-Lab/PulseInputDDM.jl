# ### Fitting a choice model
# Blah blah blah

using pulse_input_DDM

# ### Geneerate some data
# Blah blah blah

θ_syn, data = synthetic_data(;ntrials=1000)

# ### Optimize stuff
# Blah blah blah

model, options = optimize(data)

# ### Compute Hessian
# Blah blah blah

H = Hessian(model)

# ### Get the CIs from the Hessian
# Blah blah blah

CI = CIs(model, H)
