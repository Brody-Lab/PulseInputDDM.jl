
@with_kw struct Softplus_negbin{T1} <: DDMf
    r::T1 = 1e6
    c::T1 = 5.0*rand([-1,1])
end


# should eventually do what I did here for softplus and then can reuse the likelihood function
function (θ::Softplus_negbin)(x::Union{U,Vector{U}}, λ0::Union{T,Vector{T}}, dt) where {U,T <: Real}

    @unpack r, c = θ
    
    μ = softplus.(c*x .+ softplusinv.(λ0))
    p = r/(μ*dt + r)
    p = min(1. - eps(), p)
    #sig2 = μ*dt + r*(μ*dt)^2
    #p = μ*dt/sig2
    #p = exp(-log((μ*dt/r) + 1.))
    #p = 1. /((μ*dt)/r + 1.)
    
    NegativeBinomial(r, p)
    
end


"""
    all_Softplus_negbin(data)

Returns: `array` of `array` of `string`, of all Softplus_negbin
"""
function all_Softplus_negbin(data)
    
    ncells = getfield.(first.(data), :ncells)
    f = repeat(["Softplus_negbin"], sum(ncells))
    borg = vcat(0,cumsum(ncells))
    f = [f[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]
    
end


#=
function likelihood(θz, θy::Vector{Softplus_negbin{T2}}, data::neuraldata,
        P::Vector{T1}, M::Array{T1,2},
        xc::Vector{T1}, dx::T3, n, cross) where {T1,T2,T3 <: Real}

    @unpack λ, σ2_a, σ2_s, ϕ, τ_ϕ = θz
    @unpack spikes, input_data = data
    @unpack binned_clicks, clicks, dt, λ0, centered, delay, pad = input_data
    @unpack nT, nL, nR = binned_clicks
    @unpack L, R = clicks

    #adapt magnitude of the click inputs
    La, Ra = adapt_clicks(ϕ,τ_ϕ,L,R;cross=cross)

    F = zeros(T1,n,n) #empty transition matrix for time bins with clicks
    
    time_bin = (-(pad-1):nT+pad) .- delay
    
    alpha = log.(P)

    @inbounds for t = 1:length(time_bin)
        
        #mm = maximum(alpha)
        py = vcat(map(xc-> sum(map((k,θy,λ0)-> logpdf(θy(xc, λ0[t], dt), k[t]), spikes, θy, λ0)), xc)...)

        if time_bin[t] >= 1
            
            alpha, F = latent_one_step_alt!(alpha, F, λ, σ2_a, σ2_s, time_bin[t], nL, nR, La, Ra, M, dx, xc, n, dt)
            
            #=
            any(t .== nL) ? sL = sum(La[t .== nL]) : sL = zero(T1)
            any(t .== nR) ? sR = sum(Ra[t .== nR]) : sR = zero(T1)
            σ2 = σ2_s * (sL + sR);   μ = -sL + sR

            if (sL + sR) > zero(T1)
                transition_M!(F,σ2+σ2_a*dt,λ, μ, dx, xc, n, dt)
                alpha = log.((exp.(alpha .- mm)' * F')') .+ mm .+ py
            else
                alpha = log.((exp.(alpha .- mm)' * M')') .+ mm .+ py
            end
            
            
        else
            alpha = alpha .+ py
        end
            =#
        end
        
        alpha = alpha .+ py
                       
    end

    return exp(logsumexp(alpha)), exp.(alpha)/sum(exp.(alpha))

end
=#


function likelihood(θz, θy::Vector{Softplus_negbin{T2}}, data::neuraldata,
        P::Vector{T1}, M::Array{T1,2},
        xc::Vector{T1}, dx::T3, n, cross) where {T1,T2,T3 <: Real}

    @unpack λ, σ2_a, σ2_s, ϕ, τ_ϕ = θz
    @unpack spikes, input_data = data
    @unpack binned_clicks, clicks, dt, λ0, centered, delay, pad = input_data
    @unpack nT, nL, nR = binned_clicks
    @unpack L, R = clicks

    #adapt magnitude of the click inputs
    La, Ra = adapt_clicks(ϕ,τ_ϕ,L,R;cross=cross)

    F = zeros(T1,n,n) #empty transition matrix for time bins with clicks
    
    time_bin = (-(pad-1):nT+pad) .- delay
    
    c = Vector{T1}(undef, length(time_bin))

    @inbounds for t = 1:length(time_bin)

        if time_bin[t] >= 1
            P, F = latent_one_step!(P, F, λ, σ2_a, σ2_s, time_bin[t], nL, nR, La, Ra, M, dx, xc, n, dt)
        end
        
        P = P .* (vcat(map(xc-> exp(sum(map((k,θy,λ0)-> logpdf(θy(xc, λ0[t], dt), k[t]), spikes, θy, λ0))), xc)...))
        
        c[t] = sum(P)
        P /= c[t]

    end

    return c, P

end