# ### Fitting a choice model
# Blah blah blah

using pulse_input_DDM

# ### Geneerate some data
# Blah blah blah

pz, pd, data = default_parameters_and_data(ntrials=1000)

# ### Optimize stuff
# Blah blah blah

pz, pd, = optimize_model(pz, pd, data)

# ### Compute Hessian
# Blah blah blah

H = compute_Hessian(pz, pd, data; state="final")

# ### Get the CIs from the Hessian
# Blah blah blah

pz, pd = compute_CIs!(pz, pd, H)
