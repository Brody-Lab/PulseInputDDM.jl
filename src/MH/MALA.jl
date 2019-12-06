using pulse_input_DDM, Distributions

pz, pd, data = default_parameters_and_data(ntrials=5000);
x-> compute_LL(x,data)
vcat(pz["generative"],pd["generative"]),
MvNormal(zeros(9), [0.1, 1., 0.1, 1., 0.1, 0.01, 0.001, 1., 0.01])

function ∂ℓπ∂θ(x, model)
    res = GradientResult(x)
    gradient!(res, model, x)
    return (value(res), gradient(res))
end


function mala(logdensity::function, gradient::function, h, M;niter::Int=1000 ,θinit)

    θtrace = Array{Float64}(length(θinit),niter)
    θ = θinit
    θtrace[:,1] = θinit

    for i=2:niter
        θtrace[:,i]=θ
    end

    return θtrace

end

function gradientStep(θ,t)
    θ-t*M*gradient(θ)
end

function step(θ_prev, )

    #θold=θ
    ∂ℓπ∂θ(θ_prev)
    θ=rand(MvNormal(gradientStep(θ,0.5*h),h*M))


    # Generate a new proposal.
    θ = propose(spl, model, θ_prev)

    α = logdensity(θ) - logdensity(θold) + logpdf(MvNormal(gradientStep(θ,0.5*h),h*M),θold) - logpdf(MvNormal(gradientStep(θold,0.5*h),h*M),θ)

    # Decide whether to return the previous θ or the new one.
    if log(rand(rng)) < min(α, 0.0)
        return θ
    else
        return θ_prev
    end

end

q(proposal, θ::Vector{<:Real}, θcond::Vector{<:Real}) = logpdf(proposal, θ - θcond)

propose(proposal, model::DensityModel, θ::Vector{<:Real}) = Transition(model, θ + rand(proposal))

# Calculate the log acceptance probability.
α = ℓπ(model, θ) - ℓπ(model, θ_prev) + q(spl, θ_prev, θ) - q(spl, θ, θ_prev)
