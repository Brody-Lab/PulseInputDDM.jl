
#ProjDir = @__DIR__
#cd(ProjDir)
#include("LNR.jl")

#=
@model model(data, Nr) = begin
    μ = Array{Real,1}(undef,Nr)
    μ ~ [Normal(0,3)]
    s ~ Uniform(0, pi/2)
    σ = tan(s)
    #σ ~ Truncated(Cauchy(0, 1), 0, Inf)
    for i in 1:length(data)
        data[i] ~ LNR(μ, σ, 0.0)
    end
end

=#

#Random.seed!(343)
#Nr = 9
#Nobs = 10


#μ = rand(Normal(0, 3), Nr)
#σ = rand(Uniform(.2, 1))
#dist = LNR(μ=μ, σ=σ, ϕ=0.0)

using Distributions, Turing, Random, Parameters
using Base.Threads, pulse_input_DDM

pz = [eps(), 10., -0.5, 20., 1.0, 0.6, 0.02]
pd = [1.,0.05]

#data = pulse_input_DDM.sample_clicks_and_choices(pz, pd, 1000)
#data = pulse_input_DDM.bin_clicks!(data)
data = pulse_input_DDM.sample_clicks(10000)

#pz, pd, data = default_parameters_and_data(ntrials=1000);
pz2 = latent(pz...)
pd2 = choice(pd...)

dt, n = 1e-2, 53

T, L, R = data["T"], data["leftbups"], data["rightbups"]
binned = map((T,L,R)-> pulse_input_DDM.bin_clicks(T,L,R; dt=dt), data["T"], data["leftbups"], data["rightbups"])
nT, nL, nR = map(x->getindex.(binned, x), 1:3)

I = inputs.(L, R, T, nT, nL, nR, dt)
#I = inputs(L, R, T, nT, nL, nR, dt)

dist = map(i-> choiceDDM(pz2, pd2, i), I);
#dist = choiceDDM(pz2, pd2, I);

#@time LL_all_trials(pz["generative"], pd["generative"], data)
#@time pulse_input_DDM.LL_all_trials_thread(pz["generative"], pd["generative"], data)

#data = rand(dist, Nobs)
#need dtMC, not dt
choices = rand.(dist)


#logpdf.(dist, choices)

#using Distributed
#addprocs(4)

@model model(data, inputs, n, dt) = begin

    #σ2_i ~ Uniform(0., 2.)
    #B ~ Uniform(2., 30.)
    #λ ~ Uniform(-5., 5.)
    σ2_a ~ Uniform(0., 100.)
    #σ2_s ~ Uniform(0., 2.5)
    #ϕ ~ Uniform(0.01, 1.2)
    #τ_ϕ ~ Uniform(0.005, 1.)
    #bias ~ Uniform(-10., 10.)
    #lapse ~ Uniform(0., 1.)

    σ2_i = eps()
    B = 10
    λ = -0.5
    σ2_s = 1.0
    ϕ = 0.6
    τ_ϕ = 0.02
    bias = 1.0
    lapse = 0.05

    pz = latent(σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ)
    pd = choice(bias, lapse)

    Threads.@threads for i in 1:length(data)
    #@sync @distributed for i in 1:length(data)
        data[i] ~ choiceDDM(pz, pd, inputs[i])
    end
    #data ~ choiceDDM(pz, pd, inputs)
end

#chain = sample(model(choices, I), NUTS(1000, .8), 2000)
iterations = 1000
ϵ = 0.05
τ = 10
# Start sampling.
#chain = sample(coinflip(data), HMC(iterations, ϵ, τ));
#@time chain_HMC = sample(model(choices, I), HMC(ϵ, τ), iterations)
#NUTS_HMC = sample(model(choices, I), NUTS(0.65), iterations)
chain = sample(model(choices, I), MH(), 2000)
#chain = sample(model(choices, I, n, dt))

#histogram(chain[:B])
