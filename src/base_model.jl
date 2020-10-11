
const dimz = 7


"""
    CIs(H)

Given a Hessian matrix `H`, compute the 2 std confidence intervals based on the Laplace approximation.
If `H` is not positive definite (which it should be, but might not be due numerical round off, etc.) compute
a close approximation to it by adding a correction term. The magnitude of this correction is reported.

"""
function CIs(H::Array{Float64,2}) where T <: DDM

    HPSD = Matrix(cholesky(Positive, H, Val{false}))

    if !isapprox(HPSD,H)
        norm_ϵ = norm(HPSD - H)/norm(H)
        @warn "Hessian is not positive definite. Approximated by closest PSD matrix.
            ||ϵ||/||H|| is $norm_ϵ"
    end

    CI = 2*sqrt.(diag(inv(HPSD)))

    return CI, HPSD

end


"""
    P, M, xc, dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt; lapse=0.)

Creates several variables that are required to compute the LL for each trial, but that
are identical for all trials.

## PARAMETERS:

- σ2_i       initial variance

- B          bound height

- λ          drift

- σ2_a       accumlator variance

- n          number of bins

- dt         temporal bin width

## OPTIONAL PARAMETERS:

- lapse    lapse rate. Optionaly because only required for choice model.

## RETURNS:

- P    A vector. Discrete approximation to P(a).

- M    A n x n matrix. The transition matrix of P(a_t | a_{t-1})

- xc   A vector. Spatial bin centers

- dx   Scalar. The spacing between spatial bins.

## EXAMPLE CALL:

```jldoctest
```
"""
function initialize_latent_model(σ2_i::TT, B::TT, λ::TT, σ2_a::TT,
     n::Int, dt::Float64; lapse::UU=0.) where {TT,UU <: Any}

    xc,dx = bins(B,n)
    P = P0(σ2_i,n,dx,xc,dt; lapse=lapse)
    M = transition_M(σ2_a*dt,λ,zero(TT),dx,xc,n,dt)

    return P, M, xc, dx

end


"""
    P0(σ2_i, n dx, xc, dt; lapse=0.)

"""
function P0(σ2_i::TT, n::Int, dx::VV, xc::Vector{TT}, dt::Float64;
    lapse::UU=0.) where {TT,UU,VV <: Any}

    P = zeros(TT,n)
    P[ceil(Int,n/2)] = one(TT) - lapse
    P[1], P[n] = lapse/2., lapse/2.
    M = transition_M(σ2_i,zero(TT),zero(TT),dx,xc,n,dt)
    P = M * P

end


"""
    latent_one_step!(P, F, λ, σ2_a, σ2_s, t, nL, nR, La, Ra, M, dx, xc, n, dt)

"""
function latent_one_step!(P::Vector{TT}, F::Array{TT,2}, λ::TT, σ2_a::TT, σ2_s::TT,
        t::Int, nL::Vector{Int}, nR::Vector{Int},
        La::Vector{TT}, Ra::Vector{TT}, M::Array{TT,2},
        dx::UU, xc::Vector{TT}, n::Int, dt::Float64; backwards::Bool=false) where {TT,UU <: Any}

    any(t .== nL) ? sL = sum(La[t .== nL]) : sL = zero(TT)
    any(t .== nR) ? sR = sum(Ra[t .== nR]) : sR = zero(TT)

    σ2 = σ2_s * (sL + sR);   μ = -sL + sR

    if (sL + sR) > zero(TT)
        transition_M!(F,σ2+σ2_a*dt,λ, μ, dx, xc, n, dt)
        P = F * P
    else
        P = M * P
    end

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
    bins(B,n)

Computes the bin center locations and bin spacing, given the boundary and number of bins.

### Examples
```jldoctest
julia> xc,dx = pulse_input_DDM.bins(25.5,53)
([-26.0, -25.0, -24.0, -23.0, -22.0, -21.0, -20.0, -19.0, -18.0, -17.0  …  17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0], 1.0)
```
"""
function bins(B::TT, n::Int) where {TT}

    dx = 2. *B/(n-2)

    xc = vcat(collect(range(-(B+dx/2.),stop=-dx,length=Int((n-1)/2.))),0.,
        collect(range(dx,stop=(B+dx/2.),length=Int((n-1)/2))))

    return xc, dx

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
julia> dt, n, B, σ2, λ, μ = 0.1, 53, 10., 10., -0.5, 1.;

julia> xc,dx = pulse_input_DDM.bins(B, n);

julia> M = pulse_input_DDM.transition_M(σ2, λ, μ, dx, xc, n, dt);

julia> size(M)
(53, 53)
```
"""
function transition_M(σ2::TT, λ::TT, μ::TT, dx::UU,
        xc::Vector{TT}, n::Int, dt::Float64) where {TT,UU <: Any}

    M = zeros(TT,n,n)
    transition_M!(M,σ2,λ,μ,dx,xc,n,dt)

    return M

end


"""
    transition_M!(F::Array{TT,2}, σ2::TT, λ::TT, μ::TT, dx::Float64,
        xc::Vector{TT}, n::Int, dt::Float64) where {TT <: Any}

"""
function transition_M!(F::Array{TT,2}, σ2::TT, λ::TT, μ::TT, dx::UU,
        xc::Vector{TT}, n::Int, dt::Float64) where {TT,UU <: Any}

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
    adapted_clicks(ϕ, τ_ϕ, L, R)

"""
function adapt_clicks(ϕ::TT, τ_ϕ::TT, L::Vector{Float64}, R::Vector{Float64}; cross::Bool=false) where {TT}
    
    if cross
        
        all = vcat(hcat(L[2:end], -1 * ones(length(L)-1)), hcat(R, ones(length(R))))
        all = all[sortperm(all[:, 1]), :]
        adapted = ones(TT, size(all,1))
        adapted[1] = eps()

        if (typeof(ϕ) == Float64) && (isapprox(ϕ, 1.0))
        else
            (length(all) > 1 && ϕ != 1.) ? adapt_clicks!(ϕ, τ_ϕ, adapted, all[:, 1]) : nothing
        end
        
        all = vcat([0., -1.]', all)
        adapted = vcat(eps(), adapted)
        La, Ra = adapted[all[:,2] .== -1.], adapted[all[:,2] .== 1.]
        
    else

        La, Ra = ones(TT,length(L)), ones(TT,length(R))

        # magnitude of stereo clicks set to zero
        # I removed these lines on 8/8/18, because I'm not exactly sure why they are here (from Bing's original model)
        # and the cause the state to adapt even when phi = 1., which I'd like to spend time fitting simpler models to
        # check slack discussion with adrian and alex

        #if !isempty(L) && !isempty(R) && abs(L[1]-R[1]) < eps()
        La[1], Ra[1] = eps(), eps()
        #end

        if (typeof(ϕ) == Float64) && (isapprox(ϕ, 1.0))
        else
            (length(L) > 1 && ϕ != 1.) ? adapt_clicks!(ϕ, τ_ϕ, La, L) : nothing
            (length(R) > 1 && ϕ != 1.) ? adapt_clicks!(ϕ, τ_ϕ, Ra, R) : nothing
        end
        
    end

    return La, Ra

end


"""
    adapt_clicks!(Ca, C, ϕ, τ_ϕ)

"""
function adapt_clicks!(ϕ::TT, τ_ϕ::TT, Ca::Vector{TT}, C::Vector{Float64}) where {TT}

    ici = diff(C)

    for i = 1:length(ici)
        
        arg = (1/τ_ϕ) * (-ici[i] + xlogy(τ_ϕ, abs(1. - Ca[i]* ϕ)))
        
        if Ca[i]* ϕ <= 1
            Ca[i+1] = 1. - exp(arg)
        else
            Ca[i+1] = 1. + exp(arg)
        end
        
    end

end