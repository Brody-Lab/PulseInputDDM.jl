using Test, pulse_input_DDM, LinearAlgebra, Flatten, Parameters

@testset "pulse_input_DDM" begin

    n = 53

    @testset "choice_model" begin

        θ = θchoice(θz=θz(σ2_i = 0.5, B = 15., λ = -0.5, σ2_a = 50., σ2_s = 1.5,
            ϕ = 0.8, τ_ϕ = 0.05),
            bias=1., lapse=0.05)

        θ, data = synthetic_data(;θ=θ, ntrials=10, rng=1)
        model_gen = choiceDDM(θ, data)

        choices = getfield.(data, :choice);

        @test all(choices .== vcat(falses(9), trues(1)))

        @time @test round(loglikelihood(model_gen), digits=2) ≈ -3.72

        @test round(θ(data), digits=2) ≈ -3.72

        @test round(norm(gradient(model_gen)), digits=2) ≈ 14.32

        options = choiceoptions(fit = vcat(trues(9)),
            lb = vcat([0., 8., -5., 0., 0., 0.01, 0.005], [-30, 0.]),
            ub = vcat([2., 30., 5., 100., 2.5, 1.2, 1.], [30, 1.]),
            x0 = vcat([0.1, 15., -0.1, 20., 0.5, 0.8, 0.008], [0.,0.01]))

        model, = optimize(data, options; iterations=5, outer_iterations=1);
        @test round(norm(Flatten.flatten(model.θ)), digits=2) ≈ 25.05

        H = Hessian(model)
        @test round(norm(H), digits=2) ≈ 92.6

        CI, HPSD = CIs(H)

        @test round(norm(CI), digits=2) ≈ 1471.54

    end

    @testset "neural_model" begin

        f, ncells, ntrials, nparams = "Sigmoid", [1,2], [10,5], 4

        θ = θneural(θz = θz(σ2_i = 0.5, B = 15., λ = -0.5, σ2_a = 10., σ2_s = 1.2,
            ϕ = 0.6, τ_ϕ =  0.02),
            θy=[[Sigmoid() for n in 1:N] for N in ncells], ncells=ncells,
            nparams=nparams, f=f);

        data, = synthetic_data(θ, ntrials);
        model_gen = neuralDDM(θ, data);

        spikes = map(x-> sum.(x), getfield.(vcat(data...), :spikes))

        @test all(spikes .== [[3], [14], [6], [1], [5], [9], [20], [9], [5], [21], [11, 10], [5, 8], [7, 5], [10, 11], [11, 9]])

        @test round(loglikelihood(model_gen), digits=2) ≈ -529.86

        @test round(norm(gradient(model_gen)), digits=2) ≈ 52.18

        x = pulse_input_DDM.flatten(θ)
        @test round(loglikelihood(x, data, θ), digits=2) ≈ -529.86

        θy0 = vcat(vcat(θy.(data, f)...)...)
        @test round(norm(θy0), digits=2) ≈ 31.16

        #deterministic model
        options0 = Sigmoid_options_noiseless(ncells=ncells,
            fit=vcat(falses(dimz), trues(sum(ncells)*nparams)),
            x0=vcat([0., 30., 0. + eps(), 0., 0., 1. - eps(), 0.008], θy0))

        θ0 = θneural_noiseless(options0.x0, ncells, nparams, f)
        model0 = neuralDDM(θ0, data)

        @test round(loglikelihood(model0), digits=2) ≈ -1399.68

        x0 = pulse_input_DDM.flatten(θ0)
        @unpack ncells, nparams, f = θ0

        @test round(loglikelihood(x0, data, θ0), digits=2) ≈ -1399.68

        model, = optimize(data, options0; iterations=2, outer_iterations=1)
        @test round(norm(pulse_input_DDM.flatten(model.θ)), digits=2) ≈ 87.51

        @test round(norm(gradient(model)), digits=2) ≈ 6.82

        x0 = vcat([0.1, 15., -0.1, 20., 0.5, 0.8, 0.008], pulse_input_DDM.flatten(model.θ)[dimz+1:end])

        options = Sigmoid_options(ncells=ncells, x0=x0)

        model, = optimize(data, options; iterations=2, outer_iterations=1)
        @test round(norm(pulse_input_DDM.flatten(model.θ)), digits=2) ≈ 85.73

        H = Hessian(model, n; chuck_size=4)
        @test round(norm(H), digits=2) ≈ 22.83

        CI, HPSD = CIs(H)
        @test round(norm(CI), digits=2) ≈ 303.02

    end

    #add a julia file to the `test` directory with your tests
    @testset "new_changes" begin include("new_changes_tests.jl") end

end