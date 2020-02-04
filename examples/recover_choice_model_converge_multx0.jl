using pulse_input_DDM, Flatten, Random, JLD

θ_syn = θchoice(θz=θz(σ2_i = 1., B = 13., λ = -0.5, σ2_a = 10., σ2_s = 1.0,
    ϕ = 0.4, τ_ϕ = 0.02), bias=0.1, lapse=0.1);

_, data = synthetic_data(;θ=θ_syn, ntrials=5_000, rng=1, dt=2e-2);

n = 53

loglikelihood(θ_syn, data, n)

iter = parse(Int, ARGS[1])

fit = vcat(trues(7),falses(2),trues(2));
x0 = vcat([0.1, 15., -0.1, 20., 0.5, 0.2, 0.008,0.,0.], [0.,0.01])
lb=vcat([0., 8., -5., 0.,   0.,  0.01, 0.005, -30., -30.], [-5., 0.])
ub=vcat([2., 30., 5., 100., 2.5, 1.2,  1.,     30., 30.], [5., 1.]);

Random.seed!(iter)
x00 = lb + (ub - lb) .* rand(length(x0))
x00[2] = 15.
x00[8:9] .= 0.;

options = choiceoptions(x0=x00, fit=fit, 
    lb=vcat([0.,  8., -5., 0.,   0.,  0.01, 0.005, -30., -30.], [-5., 0.]),
    ub=vcat([30., 32., 5., 200., 5.,  1.2,  1.,     30.,  30.],  [5., 1.]))

model, output = optimize(data, options, n; f_tol=1e-9, extended_trace=true, show_trace=false, scaled=false)
    
fit = collect(Flatten.flatten(model.θ))
trace = hcat(map(x-> x.metadata["x"], output.trace)...)

@save "/usr/people/briandd/example_"*ARGS[1]*".jld" trace fit