n, cross, ntrials = 53, false, 2

options = neural_choice_options(f)

choice_neural_model = neural_choiceDDM(θneural_choice(vcat(x0[1:dimz], 0., 0., x0[dimz+1:end]), f), data, n, cross)

@test round(choice_loglikelihood(choice_neural_model), digits=2) ≈ -6.45

@test round(joint_loglikelihood(choice_neural_model), digits=2) ≈ -486.23

import PulseInputDDM: nθparams
nparams, = nθparams(f)

fit = vcat(falses(dimz), trues(2), falses.(nparams)...);
options = neural_choice_options(fit=fit, lb=options.lb, ub=options.ub)

choice_neural_model, = choice_optimize(choice_neural_model, options; iterations=2, outer_iterations=1)

@test round(norm(PulseInputDDM.flatten(choice_neural_model.θ)), digits=2) ≈ 42.06

choice_neural_model = neural_choiceDDM(θneural_choice(vcat(x0[1:dimz], 0., 0., x0[dimz+1:end]), f), data, n, cross)

fit = vcat(trues(dimz), trues(2), trues.(nparams)...);
options = neural_choice_options(fit=fit, lb=vcat(options.lb[1:7], -10., options.lb[9:end]), 
    ub=vcat(options.ub[1:7], 10., options.ub[9:end]))

choice_neural_model, = choice_optimize(choice_neural_model, options; iterations=2, outer_iterations=1)

@test round(norm(PulseInputDDM.flatten(choice_neural_model.θ)), digits=2) ≈ 42.068