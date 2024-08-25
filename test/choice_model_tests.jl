ntrials = 2

θ = θchoice(θz=θz(σ2_i = 0.5, B = 15., λ = -0.5, σ2_a = 50., σ2_s = 1.5,
    ϕ = 0.8, τ_ϕ = 0.05),
    bias=1., lapse=0.05)

θ, data = synthetic_data(;θ=θ, ntrials=ntrials, rng=1)
model_gen = choiceDDM(θ=θ)

choices = getfield.(data, :choice)

@test all(choices .== vcat(true, false))  
        
@time @test round(loglikelihood(model_gen, data), digits=2) ≈ -0.79

@test round(norm(gradient(model_gen, data)), digits=2) ≈ 0.67
       
options = choiceoptions(lb=vcat([0., 8.,  -5., 0.,   0.,  0.01, 0.005], [-30, 0.]),
    ub = vcat([2., 30., 5., 100., 2.5, 1.2,  1.], [30, 1.]), 
    fit = trues(dimz+2));

x0 = vcat([0.1, 15., -0.1, 20., 0.5, 0.8, 0.008], [0.,0.01])
θ = Flatten.reconstruct(θchoice(), x0)
model = choiceDDM(θ=θ)

model, = fit(model, data, options; iterations=5, outer_iterations=1);
@test round(norm(Flatten.flatten(model.θ)), digits=2) ≈ 25.03

H = Hessian(model, data)
@test round(norm(H), digits=2) ≈ 7.62

CI, HPSD = CIs(H)

@test round(norm(CI), digits=2) ≈ 1503.7