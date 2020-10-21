# # Loading data and fitting a choice model

using pulse_input_DDM, Flatten
data = load_choice_data("/scratch/ejdennis/bingdata/chrono_B052_rawdata.mat");
n = 53

options = choiceoptions(fit = vcat(trues(9),
    lb = vcat([0., 1., -5., 0., 0., 0.01, 0.005, -30.0, 0.0]),
    ub = vcat([2., 30., 5., 100., 100, 1.2, 1., 30.0, 1.0]))

x0 = vcat([0.128, 6., -0.386, 0.0272, 36.6, 0.154, 0.14], [-0.18,0.094]);
save_file = "/scratch/ejdennis/ddm_runs/b052_results.mat"
θ = Flatten.reconstruct(θchoice(), x0)

model, = optimize(θ, data, options; iterations=1, outer_iterations=1)
H = Hessian(model)
CI, = CIs(H);
save_choice_model(save_file, model, options, CI)
