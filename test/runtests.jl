using Test, pulse_input_DDM, LinearAlgebra, Flatten, Parameters

@testset "pulse_input_DDM" begin

    n, cross, initpt_mod = 53, false, false
    
    @testset "choice_model" begin

        θ = θchoice(θz=θz(σ2_i = 0.5, B = 15., λ = -0.5, σ2_a = 50., σ2_s = 1.5,
                    ϕ = 0.8, τ_ϕ = 0.05), bias=1., 
                    θlapse=θlapse(lapse_prob=0.05,lapse_bias=0., lapse_modbeta=0.), 
                    θhist=θtrialhist(h_βc = 0., h_βe= 0., h_ηc =0., h_ηe= 0.))

        θ, data = synthetic_data(;θ=θ, ntrials=10, rng=1, initpt_mod=initpt_mod)
        model_gen = choiceDDM(θ, data, n, cross, initpt_mod, θprior(μ_B=40., σ_B=1e6))

        choices = getfield.(data, :choice)

        @test all(choices .== vcat(falses(9), trues(1)))

        @time @test round(loglikelihood(model_gen), digits=2) ≈ -3.72

        @test round(norm(gradient(model_gen)), digits=2) ≈ 14.32

        options, x0 = create_options_and_x0()    
        model, = optimize(data, options; iterations=5, outer_iterations=1, 
            θprior=θprior(μ_B=40., σ_B=1e6));
        @test round(norm(Flatten.flatten(model.θ)), digits=2) ≈ 25.05

        H = Hessian(model)
        @test round(norm(H), digits=2) ≈ 92.6

        CI, HPSD = CIs(H)

        @test round(norm(CI), digits=2) ≈ 1471.54

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

        @test all(spikes .== [[3], [14], [6], [1], [5], [9], [20], [9], [5], [21], [11, 10], [5, 8], [7, 5], [10, 11], [11, 9]])

        @test round(loglikelihood(model_gen), digits=2) ≈ -529.86

        @test round(norm(gradient(model_gen)), digits=2) ≈ 52.18

        x = pulse_input_DDM.flatten(θ)
        @test round(loglikelihood(x, model_gen), digits=2) ≈ -529.86

        θy0 = vcat(vcat(θy.(data, f)...)...)
        @test round(norm(θy0), digits=2) ≈ 31.16

        options0 = neural_options_noiseless(f)
        x0=vcat([0., 30., 0. + eps(), 0., 0., 1. - eps(), 0.008], θy0)

        θ0 = θneural_noiseless(x0, f)
        model0 = noiseless_neuralDDM(θ0, data)

        @test round(loglikelihood(model0), digits=2) ≈ -1399.68

        x0 = pulse_input_DDM.flatten(θ0)
        @unpack f = θ0

        @test round(loglikelihood(x0, model0), digits=2) ≈ -1399.68

        model, = optimize(model0, options0; iterations=2, outer_iterations=1)
        @test round(norm(pulse_input_DDM.flatten(model.θ)), digits=2) ≈ 87.51

        @test round(norm(gradient(model)), digits=2) ≈ 6.82

        x0 = vcat([0.1, 15., -0.1, 20., 0.5, 0.8, 0.008], pulse_input_DDM.flatten(model.θ)[dimz+1:end])
        options = neural_options(f)  

        model = neuralDDM(θneural(x0, f), data, n, cross, θprior(μ_B=40., σ_B=1e6))
        model, = optimize(model, options; iterations=2, outer_iterations=1)
        @test round(norm(pulse_input_DDM.flatten(model.θ)), digits=2) ≈ 85.73

        H = Hessian(model; chuck_size=4)
        @test round(norm(H), digits=2) ≈ 22.83

        CI, HPSD = CIs(H)
        @test round(norm(CI), digits=2) ≈ 303.03

    end

    #add a julia file to the `test` directory with your tests
    @testset "new_changes" begin include("new_changes_tests.jl") end

end