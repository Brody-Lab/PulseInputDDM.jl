using AdvancedMH, pulse_input_DDM, Distributions, Random
using LinearAlgebra
using PyPlot

#increase support?
#step size?
#spl = MetropolisHastings(vcat(pz["generative"],pd["generative"]), MvNormal(zeros(9), [0.1, 1., 0.1, 1., 0.1, 0.01, 0.001, 1., 0.01]));
#spl = MetropolisHastings(vcat(pz["generative"],pd["generative"]), MvNormal(zeros(9), [0.1, 1., 0.1, 1., 0.1, 0.1, 0.01, 1., 0.01]));

pz, pd, data = default_parameters_and_data(ntrials=5000)
@everywhere ℓπ(x) = compute_LL(x,data)
model = DensityModel(ℓπ)

#using PositiveFactorizations

#h = compute_Hessian(pz, pd, data; state="generative");
#h⁻¹ = inv(-1*h)
#F⁻¹ = Matrix(cholesky(Positive, h⁻¹, Val{false}))
#F = Matrix(cholesky(Positive, -1*h, Val{false}))

using AdvancedHMC

θinit = vcat(pz["initial"],pd["initial"])

∂ℓπ∂θ(θ) = AdvancedMH.∂ℓπ∂θ(model, θ)
metric = DiagEuclideanMetric(length(θinit))
h = Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
τ = find_good_eps(h, θinit)
#τ = τ^2
#τ = 1/(eigvals(F)[end])
#τ = 0.003^2
#change support
#M = F⁻¹

#h = 0.5 * (h + h')
#V = eigvecs(h⁻¹)
#d = collect(Diagonal(max.(0, eigvals(h⁻¹))))
#h⁻¹PSD = V * d * V'
#h⁻¹PSD = 0.5 * (h⁻¹PSD + h⁻¹PSD')
#hPSD⁻¹ = inv(hPSD)
#h⁻¹ = 0.5 * (h⁻¹ + h⁻¹')

nsamples = 5000

#τ = 0.0001
#if i remember, when I did this, things were better, probably because wasn't being squared or rooted or whatever.
#M = Diagonal([0.1, 1., 0.1, 1., 0.1, 0.01, 0.001, 1., 0.01])
M = I(9)
spl_MALA = MALA(θinit, x-> MvNormal(τ^2 * 0.5 * M * x, τ^2 * M))
@time chain_MALA_2 = sample(Random.GLOBAL_RNG, model, spl_MALA, nsamples;
    param_names=vcat(pz["name"], pd["name"]))

spl = MetropolisHastings(θinit, MvNormal(zeros(9), 5e-5 * M))

#gelmandiag(c::AbstractChains; alpha=0.05, mpsrf=false, transform=false)


#add back mapping functions?
@time chain_MH_6 = sample(Random.GLOBAL_RNG, model, spl, nsamples; param_names=vcat(pz["name"], pd["name"]))

for i = 1:9
    subplot(3,3,i)
    plot(vec(chain_MH_3[chain_MH.name_map.parameters[i]].value), color="red")
    plot(vec(chain_MH_4[chain_MH.name_map.parameters[i]].value), color="blue")
    plot(vec(chain_MH_5[chain_MH.name_map.parameters[i]].value), color="orange")
    plot(vec(chain_MH_2[chain_MH.name_map.parameters[i]].value), color="black")
    plot(vec(chain_MH_6[chain_MH.name_map.parameters[i]].value), color="yellow")
    plot(vec(chain_MALA[chain_MALA.name_map.parameters[i]].value), color="purple")
    plot(vec(chain_MALA_2[chain_MALA_2.name_map.parameters[i]].value), color="green")
    title(chain_MALA.name_map.parameters[i])
end

tight_layout()

1. - sum(isapprox.(sum(diff(chain_MH_4.value.data[:,findall(chain_MH_4.value.axes[2] .!= "lp"),1],
    dims=1), dims=2), 0.)) / nsamples

1. - sum(isapprox.(sum(diff(chain_MH.value.data[:,findall(chain_MH.value.axes[2] .!= "lp"),1],
    dims=1), dims=2), 0.)) / nsamples

1. - sum(isapprox.(sum(diff(chain_MH_5.value.data[:,findall(chain_MH_5.value.axes[2] .!= "lp"),1],
    dims=1), dims=2), 0.)) / nsamples

1. - sum(isapprox.(sum(diff(chain_MH_3.value.data[:,findall(chain_MH_3.value.axes[2] .!= "lp"),1],
    dims=1), dims=2), 0.)) / nsamples

1. - sum(isapprox.(sum(diff(chain_MH_6.value.data[:,findall(chain_MH_6.value.axes[2] .!= "lp"),1],
    dims=1), dims=2), 0.)) / nsamples

1. - sum(isapprox.(sum(diff(chain_MH_2.value.data[:,findall(chain_MH_2.value.axes[2] .!= "lp"),1],
    dims=1), dims=2), 0.)) / nsamples

1. - sum(isapprox.(sum(diff(chain_MALA.value.data[:,findall(chain_MALA.value.axes[2] .!= "lp"),1],
    dims=1), dims=2), 0.)) / nsamples

1. - sum(isapprox.(sum(diff(chain_MALA_2.value.data[:,findall(chain_MALA_2.value.axes[2] .!= "lp"),1],
    dims=1), dims=2), 0.)) / nsamples
