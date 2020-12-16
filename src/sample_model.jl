"""
    synthetic_clicks(ntrials, rng)

Computes randomly timed left and right clicks for ntrials.
rng sets the random seed so that clicks can be consistently produced.
Output is bundled into an array of 'click' types.
"""
function synthetic_clicks(ntrials::Int, rng::Int;
    tmin::Float64=0.2, tmax::Float64=1.0, clickrate::Int=40)

    Random.seed!(rng)

    T = tmin .+ (tmax-tmin).*rand(ntrials)
    T = ceil.(T, digits=2)
    clicktot = round.(Int, clickrate.*T)

    rate_vals = [15.10, 24.89, 7.29, 32.7, 3.03, 36.96, 1.17, 38.82]
    Rbar = rand(rate_vals, ntrials)
    Lbar = clickrate .- Rbar    

    R = cumsum.(rand.(Exponential.(1 ./Rbar), clicktot))
    L = cumsum.(rand.(Exponential.(1 ./Lbar), clicktot))

    R = map((T,R)-> vcat(0,R[R .<= T]), T,R)
    L = map((T,L)-> vcat(0,L[L .<= T]), T,L)

    clicks.(L, R, T)

end


"""
    rand(θz, inputs)

Generate a sample latent trajectory, given parameters of the latent model `θz` and `inputs` for one trial.
if `a_0` is provided, that sets the initial values of the latent. Default value for `a_0` is 0

Returns:

- `A`: an `array` of the latent path.

"""
function rand(θz::θz{T}, inputs; a_0::Float64 = 0.) where T <: Real

    @unpack σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ = θz
    @unpack clicks, binned_clicks, centered, dt, delay, pad = inputs
    @unpack nT, nL, nR = binned_clicks
    @unpack L, R = clicks

    La, Ra = adapt_clicks(ϕ, τ_ϕ, L, R)

    time_bin = (-(pad-1):nT+pad) .- delay
    
    A = Vector{T}(undef, length(time_bin))
    
    if σ2_i > 0.
        a = sqrt(σ2_i)*randn() + a_0
    else
        a = zero(typeof(σ2_i)) + a_0
    end

    for t = 1:length(time_bin)
        
         if time_bin[t] < 1
                    
            if σ2_i > 0.
                a = sqrt(σ2_i)*randn() + a_0
            else
                a = zero(typeof(σ2_i)) + a_0
            end
            
        else
            
            a = sample_one_step!(a, time_bin[t], σ2_a, σ2_s, λ, nL, nR, La, Ra, dt)
            
        end

        abs(a) > B ? (a = B * sign(a); A[t:end] .= a; break) : A[t] = a

    end

    return A

end


"""
    sample_one_step!(a, t, σ2_a, σ2_s, λ, nL, nR, La, Ra, dt)

Move latent state one dt forward, given parameters defining the DDM.
"""
function sample_one_step!(a::TT, t::Int, σ2_a::TT, σ2_s::TT, λ::TT,
        nL::Vector{Int}, nR::Vector{Int},
        La, Ra, dt::Float64) where {TT <: Any}

    any(t .== nL) ? sL = sum(La[t .== nL]) : sL = zero(TT)
    any(t .== nR) ? sR = sum(Ra[t .== nR]) : sR = zero(TT)
    σ2, μ = σ2_s * (sL + sR), -sL + sR


    if (σ2_a * dt + σ2) > 0.
        η = sqrt(σ2_a * dt + σ2) * randn()
    else
        η = zero(typeof(σ2_a))
    end

    #if abs(λ) < 1e-150
    #    a += μ + η
    #else
    #    h = μ/(dt*λ)
    #    a = exp(λ*dt)*(a + h) - h + η
    #end
    
    a = exp(λ*dt)*a + μ * expm1_div_x(λ*dt) + η

    return a

end



# """
# """
# function rand(θz, inputs, P::Vector{TT}, M::Array{TT,2}, n::Int,
#         xc::Vector{TT}, dx::Float64, cross::Bool) where {TT,UU <: Real}

#     @unpack λ,σ2_a,σ2_s,ϕ,τ_ϕ = θz
#     @unpack binned_clicks, clicks, dt = inputs
#     @unpack nT, nL, nR = binned_clicks
#     @unpack L, R = clicks

#     La, Ra = adapt_clicks(ϕ,τ_ϕ,L,R; cross=cross)
#     F = zeros(TT,n,n)
#     a = Vector{TT}(undef,nT)

#     @inbounds for t = 1:nT

#         P,F = latent_one_step!(P,F,λ,σ2_a,σ2_s,t,nL,nR,La,Ra,M,dx,xc,n,dt)
        
#         P /= sum(P)
        
#         a[t] = xc[findfirst(cumsum(P) .> rand())]
#         P = TT.(xc .== a[t])
        
#     end

#     return a
    
# end


# """
# """
# function randP(θz, inputs, P::Vector{TT}, M::Array{TT,2}, n::Int, 
#         xc::Vector{TT}, dx::Float64, cross::Bool) where {TT,UU <: Real}

#     @unpack λ,σ2_a,σ2_s,ϕ,τ_ϕ = θz
#     @unpack binned_clicks, clicks, dt = inputs
#     @unpack nT, nL, nR = binned_clicks
#     @unpack L, R = clicks

#     La, Ra = adapt_clicks(ϕ,τ_ϕ,L,R; cross=cross)
#     F = zeros(TT,n,n)

#     @inbounds for t = 1:nT

#         P,F = latent_one_step!(P,F,λ,σ2_a,σ2_s,t,nL,nR,La,Ra,M,dx,xc,n,dt)        
#         P /= sum(P)
  
        
#     end

#     return P
    
# end