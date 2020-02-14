using pulse_input_DDM

path = ENV["HOME"]*"/Projects/neural_DDM_analysis/data/hanks_data_sessions"

ratnames = ["B068","T034","T036","T063","T030", "T035","T011","B053", "T080","T103","E021"]
rat = ratnames[parse(Int, ARGS[1])]
sessions = filter(x->occursin(rat,x), readdir(path))

output = load.(joinpath.(path, sessions), false, delay=0.0)

data = getindex.(output, 1)

f, ncells, ntrials, nparams = "Sigmoid", map(x-> x[1].ncells, data), length.(data), 4

dt, n = 1e-2, 53

save_file = ENV["HOME"]*"/Projects/pulse_input_DDM/examples/bdd_work/results_data/latent/"*rat*".mat"

if isfile(save_file)
    x0 = reload(save_file)
    
else
    
    θy0 = vcat(vcat(initialize_θy.(data, f)...)...)
    
    options0 = neuraloptions(ncells=ncells,
        fit=vcat(falses(dimz), trues(sum(ncells)*nparams)),
        x0=vcat([0., 30., 0. + eps(), 0., 0., 1. - eps(), 0.008], θy0))

    model, = optimize(data, options0; show_trace=false)
    
    fit=vcat(falses(1), trues(dimz-1), trues(sum(ncells)*nparams))
    x0=vcat([0.1, 12., -2., 10., 0.1, 0.8, 0.008], pulse_input_DDM.flatten(model.θ)[dimz+1:end])
    
end

options = neuraloptions(ncells=ncells, x0=x0, fit=fit, nparams=nparams, f=f)

model, = optimize(data, options, n)

H = Hessian(model, n)
CI, HPSD = CIs(H)

save(save_file, model, options, CI)