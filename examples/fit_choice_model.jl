# # Fitting a choice model to loaded data
# Blah blah blah

using pulse_input_DDM

# ### Load some data
# Blah blah blah

data = load("./examples/example_matfile.mat")

# ### Set options for optimization
# Blah blah blah

n = 53

options = choiceoptions(fit = vcat(trues(9)),
    lb = vcat([0., 8., -5., 0., 0., 0.01, 0.005], [-30, 0.]),
    ub = vcat([2., 30., 5., 100., 2.5, 1.2, 1.], [30, 1.]),
    x0 = vcat([0.1, 15., -0.1, 20., 0.5, 0.8, 0.008], [0.,0.01]))

# ### Load some data
# Blah blah blah
save_file = "./examples/example_results.mat"

#if you've already ran the optimization once and want to restart from where you stoped, this will reload those parameters
if isfile(save_file)
    options.x0 = reload(save_file)
end

# ### Optimize stuff
# Blah blah blah

model = optimize(data, options, n; iterations=5, outer_iterations=1)

# ### Compute Hessian and the confidence interavls
# Blah blah blah

H = Hessian(model, n)
CI, HPSD = CIs(model, H);

# ### Save results
# Blah blah blah

save(save_file, model, options, CI)
