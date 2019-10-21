
const dimz = 7

"""
    initialize_latent_model(pz::Vector{TT}, dx::Float64, dt::Float64;
        L_lapse::TT=0., R_lapse::TT=0.) where {TT <: Any}

"""
function initialize_latent_model(pz::Vector{TT}, dx::Float64, dt::Float64;
        L_lapse::TT=0., R_lapse::TT=0.) where {TT <: Any}

    #break up latent variables
    σ2_i,B,λ,σ2_a = pz[1:4]

    #bin centers and number of bins
    xc,n = bins(B,dx)

    # make initial latent distribution
    P = P0(σ2_i,n,dx,xc,dt; L_lapse=L_lapse, R_lapse=R_lapse)

    # build state transition matrix for times when there are no click inputs
    M = transition_M(σ2_a*dt,λ,zero(TT),dx,xc,n,dt)

    return P, M, xc, n

end



"""
    P0(σ2_i::TT, n::Int, dx::Float64, xc::Vector{TT}, dt::Float64;
        L_lapse::TT=0., R_lapse::TT=0.) where {TT <: Any}

"""
function P0(σ2_i::TT, n::Int, dx::Float64, xc::Vector{TT}, dt::Float64;
        L_lapse::TT=0., R_lapse::TT=0.) where {TT <: Any}

    P = zeros(TT,n)
    # make initial delta function
    P[ceil(Int,n/2)] = one(TT) - (L_lapse + R_lapse)
    P[1], P[n] = L_lapse, R_lapse
    M = transition_M(σ2_i,zero(TT),zero(TT),dx,xc,n,dt)
    P = M * P

end



"""
    latent_one_step!(P::Vector{TT}, F::Array{TT,2}, pz::Vector{TT}, t::Int,
        nL::Vector{Int}, nR::Vector{Int},
        La::Vector{TT}, Ra::Vector{TT}, M::Array{TT,2},
        dx::Float64, xc::Vector{TT}, n::Int, dt::Float64; backwards::Bool=false) where {TT <: Any}

"""
function latent_one_step!(P::Vector{TT}, F::Array{TT,2}, pz::Vector{TT}, t::Int,
        nL::Vector{Int}, nR::Vector{Int},
        La::Vector{TT}, Ra::Vector{TT}, M::Array{TT,2},
        dx::Float64, xc::Vector{TT}, n::Int, dt::Float64; backwards::Bool=false) where {TT <: Any}

    λ, σ2_a, σ2_s = pz[3:5]

    any(t .== nL) ? sL = sum(La[t .== nL]) : sL = zero(TT)
    any(t .== nR) ? sR = sum(Ra[t .== nR]) : sR = zero(TT)

    σ2 = σ2_s * (sL + sR);   μ = -sL + sR

    (sL + sR) > zero(TT) ? (transition_M!(F,σ2+σ2_a*dt,λ, μ, dx, xc, n, dt); P  = F * P;) : P = M * P

    #=
    if backwards
        (sL + sR) > zero(TT) ? (M!(F,σ2+σ2_a*dt,λ, μ/dt, dx, xc, n, dt); P  = F' * P;) : P = M' * P
    else
        (sL + sR) > zero(TT) ? (M!(F,σ2+σ2_a*dt,λ, μ/dt, dx, xc, n, dt); P  = F * P;) : P = M * P
    end
    =#

    return P, F

end



"""
    bins(B,dx)

Computes the bin center locations and number of bins, given the boundary and desired (average) bin spacing.

### Examples
```jldoctest
julia> xc,n = pulse_input_DDM.bins(10.,0.25)
([-10.25, -9.75, -9.5, -9.25, -9.0, -8.75, -8.5, -8.25, -8.0, -7.75  …  7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 9.75, 10.25], 81)
```
"""
function bins(B::TT,dx::Float64) where {TT <: Any}

    xc = collect(0.:dx:floor(value(B)/dx)*dx)

    if xc[end] == B
        xc = vcat(xc[1:end-1], B + dx)
    else
        xc = vcat(xc, 2*B - xc[end])
    end

    xc = vcat(-xc[end:-1:2], xc)
    n = length(xc)

    return xc, n

end



"""
    expm1_div_x(x)

"""
function expm1_div_x(x)

    y = exp(x)
    y == 1. ? one(y) : (y-1.)/log(y)

end





"""
    transition_M(σ2, λ, μ, dx, xc, n, dt)

Returns a \$n \\times n\$ Markov transition matrix. The transition matrix is discrete approximation to the Fokker-Planck equation with drift λ, diffusion σ2 and driving current (i.e. click input) μ. dx and dt define the spatial and temporal binning, respectively. xc are the bin center locations.

See also: [`transition_M!`](@ref)

### Examples
```jldoctest
julia> dt, dx, B, σ2, λ, μ = 0.1, 0.25, 10., 10., -0.5, 1.;

julia> xc,n = pulse_input_DDM.bins(B, dx);

julia> M = pulse_input_DDM.transition_M(σ2, λ, μ, dx, xc, n, dt);

julia> size(M)
(81, 81)
```
"""
function transition_M(σ2::TT, λ::TT, μ::TT, dx::Float64,
        xc::Vector{TT}, n::Int, dt::Float64) where {TT <: Any}

    M = zeros(TT,n,n)
    transition_M!(M,σ2,λ,μ,dx,xc,n,dt)

    return M

end


"""
    transition_M!(F::Array{TT,2}, σ2::TT, λ::TT, μ::TT, dx::Float64,
        xc::Vector{TT}, n::Int, dt::Float64) where {TT <: Any}

"""
function transition_M!(F::Array{TT,2}, σ2::TT, λ::TT, μ::TT, dx::Float64,
        xc::Vector{TT}, n::Int, dt::Float64) where {TT <: Any}

    F[1,1] = one(TT); F[n,n] = one(TT); F[:,2:n-1] = zeros(TT,n,n-2)

    ndeltas = max(70,ceil(Int, 10. *sqrt(σ2)/dx))

    deltaidx = collect(-ndeltas:ndeltas)
    deltas = deltaidx * (5. *sqrt(σ2))/ndeltas
    ps = exp.(-0.5 * (5*deltaidx./ndeltas).^2)
    ps = ps/sum(ps)

    @inbounds for j = 2:n-1

        #abs(λ) < 1e-150 ? mu = xc[j] + μ : mu = exp(λ*dt)*(xc[j] + μ/(λ*dt)) - μ/(λ*dt)
        #abs(λ) < 1e-150 ? mu = xc[j] + h * dt : mu = exp(λ*dt)*(xc[j] + h/λ) - h/λ
        #mu = exp(λ*dt)*xc[j] + μ * (exp(λ*dt) - 1.)/(λ*dt)
        #mu = exp(λ*dt)*xc[j] + μ * (expm1(λ*dt)/(λ*dt)
        mu = exp(λ*dt)*xc[j] + μ * expm1_div_x(λ*dt)

        #now we're going to look over all the slices of the gaussian
        for k = 1:2*ndeltas+1

            s = mu + deltas[k]

            if s <= xc[1]

                F[1,j] += ps[k]

            elseif s >= xc[n]

                F[n,j] += ps[k]

            else

                if (xc[1] < s) && (xc[2] > s)

                    lp,hp = 1,2

                elseif (xc[n-1] < s) && (xc[n] > s)

                    lp,hp = n-1,n

                else

                    hp,lp = ceil(Int, (s-xc[2])/dx) + 2, floor(Int, (s-xc[2])/dx) + 2

                end

                if hp == lp

                    F[lp,j] += ps[k]

                else

                    dd = xc[hp] - xc[lp]
                    F[hp,j] += ps[k]*(s-xc[lp])/dd
                    F[lp,j] += ps[k]*(xc[hp]-s)/dd

                end

            end

        end

    end

end


"""
    make_adapted_clicks(pz::Vector{TT}, L::Vector{Float64}, R::Vector{Float64}) where {TT}

"""
function make_adapted_clicks(pz::Vector{TT}, L::Vector{Float64}, R::Vector{Float64}) where {TT}

    ϕ,τ_ϕ = pz[6:7]

    La, Ra = ones(TT,length(L)), ones(TT,length(R))

    # magnitude of stereo clicks set to zero
    # I removed these lines on 8/8/18, because I'm not exactly sure why they are here (from Bing's original model)
    # and the cause the state to adapt even when phi = 1., which I'd like to spend time fitting simpler models to
    # check slack discussion with adrian and alex

    #if !isempty(L) && !isempty(R) && abs(L[1]-R[1]) < eps()
    #    La[1], Ra[1] = eps(), eps()
    #end

    (length(L) > 1 && ϕ != 1.) ? (ici_L = diff(L); adapt_clicks!(La, ϕ, τ_ϕ, ici_L)) : nothing
    (length(R) > 1 && ϕ != 1.) ? (ici_R = diff(R); adapt_clicks!(Ra, ϕ, τ_ϕ, ici_R)) : nothing

    return La, Ra

end





"""
    adapt_clicks!(Ca::Vector{TT},  ϕ::TT, τ_ϕ::TT, ici::Vector{Float64}) where {TT}

"""
function adapt_clicks!(Ca::Vector{TT},  ϕ::TT, τ_ϕ::TT, ici::Vector{Float64}) where {TT}

    for i = 1:length(ici)
        arg = xlogy(τ_ϕ, abs(1. - Ca[i]* ϕ))
        Ca[i+1] = 1. - exp((-ici[i] + arg)/τ_ϕ)
    end

end
