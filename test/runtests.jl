using Test, PulseInputDDM, LinearAlgebra, Flatten, Parameters

@testset "PulseInputDDM" begin

    n, cross = 53, false
    
    @testset "choice_model" begin

        θ = θchoice(θz=θz(σ2_i = 0.5, B = 15., λ = -0.5, σ2_a = 50., σ2_s = 1.5,
            ϕ = 0.8, τ_ϕ = 0.05),
            bias=1., lapse=0.05)

        θ, data = synthetic_data(;θ=θ, ntrials=10, rng=1)
        model_gen = choiceDDM(θ, data, n, cross, θprior(μ_B=40., σ_B=1e6))

        choices = getfield.(data, :choice)

        @test all(choices .== vcat(true, falses(8), true))  
        
        @time @test round(loglikelihood(model_gen), digits=2) ≈ -3.3

        @test round(norm(gradient(model_gen)), digits=2) ≈ 6.27
        
        options = choiceoptions(lb=vcat([0., 8.,  -5., 0.,   0.,  0.01, 0.005], [-30, 0.]),
            ub = vcat([2., 30., 5., 100., 2.5, 1.2,  1.], [30, 1.]), 
            fit = trues(dimz+2));

        model, = optimize(data, options; iterations=5, outer_iterations=1, 
            θprior=θprior(μ_B=40., σ_B=1e6));
        @test round(norm(Flatten.flatten(model.θ)), digits=2) ≈ 25.01

        H = Hessian(model)
        @test round(norm(H), digits=2) ≈ 762.91

        CI, HPSD = CIs(H)

        @test round(norm(CI), digits=2) ≈ 587.96

    end

    @testset "neural_model" begin

        ncells, ntrials = [1,2], [10,5]
        f = [repeat(["Sigmoid"], N) for N in ncells]

        θ = θneural(θz = θz(σ2_i = 0.5, B = 15., λ = -0.5, σ2_a = 10., σ2_s = 1.2,
            ϕ = 0.6, τ_ϕ =  0.02),
            θy=[[Sigmoid() for n in 1:N] for N in ncells], f=f);

        data, = synthetic_data(θ, ntrials, ncells);
        model_gen = neuralDDM(θ, data, n, cross, θprior(μ_B=40., σ_B=1e6));

        spikes = map(x-> sum.(x), getfield.(vcat(data...), :spikes))

        @test all(spikes .== [[7], [8], [5], [14], [10], [10], [9], [5], [8], [5], [4, 3], [9, 13], [11, 8], [1, 3], [11, 10]])        
        @test round(loglikelihood(model_gen), digits=2) ≈ -451.56

        @test round(norm(gradient(model_gen)), digits=2) ≈ 4.4

        x = PulseInputDDM.flatten(θ)
        @test round(loglikelihood(x, model_gen), digits=2) ≈ -451.56

        θy0 = vcat(vcat(θy.(data, f)...)...)
        @test round(norm(θy0), digits=2) ≈ 21.41

        options0 = neural_options_noiseless(f)
        x0=vcat([0., 30., 0. + eps(), 0., 0., 1. - eps(), 0.008], θy0)

        θ0 = θneural_noiseless(x0, f)
        model0 = noiseless_neuralDDM(θ0, data)

        @test round(loglikelihood(model0), digits=2) ≈ -1127.15

        x0 = pulse_input_DDM.flatten(θ0)
        @unpack f = θ0

        @test round(loglikelihood(x0, model0), digits=2) ≈ -1127.15

        model, = optimize(model0, options0; iterations=2, outer_iterations=1)
        @test round(norm(PulseInputDDM.flatten(model.θ)), digits=2) ≈ 45.21

        #=
        @test round(norm(gradient(model)), digits=2) ≈ 4.64
        =#

        x0 = vcat([0.1, 15., -0.1, 20., 0.5, 0.8, 0.008], pulse_input_DDM.flatten(model.θ)[dimz+1:end])
        options = neural_options(f)  

        model = neuralDDM(θneural(x0, f), data, n, cross, θprior(μ_B=40., σ_B=1e6))
        model, = optimize(model, options; iterations=2, outer_iterations=1)
        @test round(norm(PulseInputDDM.flatten(model.θ)), digits=2) ≈ 41.17

        H = Hessian(model; chunk_size=4)
        @test round(norm(H), digits=2) ≈ 9.17

        CI, HPSD = CIs(H)
        @test round(norm(CI), digits=2) ≈ 917.8        
        
        #joint
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

        @test round(norm(PulseInputDDM.flatten(choice_neural_model.θ)), digits=2) ≈ 52.21
        

    end

end
