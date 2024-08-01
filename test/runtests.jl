using Test, Random, PulseInputDDM, LinearAlgebra, Flatten, Parameters

@testset "PulseInputDDM" begin

    #check that random number generator from Base julia has not changed
    Random.seed!(1)
    @test isapprox(sum(randn(10)), 2.86; atol=0.01)
    
    @testset "choice_model" begin

        include("choice_model_tests.jl")

    end

    @testset "neural_model" begin

        include("neural_model_tests.jl")

    end

    @testset "joint_model" begin

        include("joint_model_tests.jl")

    end

end
