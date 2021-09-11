"""
    initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt)

"""
function initialize_latent_model(σ2_i::TT, B0::TT, λ::TT, σ2_a::TT,
     dx::Float64, dt::Float64, a_0::TT) where {TT,UU <: Any}

    xc,n = bins(B0,dx)
    P = P0(σ2_i,n,a_0,dx,xc,dt)

    return P, xc, n

end


"""
    P0(σ2_i, n dx, xc, dt)

"""
function P0(σ2_i::TT, n::Int, a_0::TT, dx::Float64, xc::Vector{TT}, dt::Float64) where {TT,UU,VV <: Any}

    P = zeros(TT,n)
    P[ceil(Int,n/2)] = one(TT) 
    M = transition_M(σ2_i,zero(TT),a_0,dx,xc,n,dt)
    P = M * P

end


"""
    latent_one_step!(P, F, λ, σ2_a, σ2_s, t, nL, nR, La, Ra, M, dx, xc, n, dt)
    for when bound is stationary

"""
function latent_one_step!(P::Vector{TT}, F::Array{TT,2}, λ::TT, σ2_a::TT, σ2_s::TT,
        t::Int, nL::Vector{Int}, nR::Vector{Int},
        La::Vector{TT}, Ra::Vector{TT}, scaled_a_0::TT,
        dx::Float64, xc::Vector{TT}, n::Int, dt::Float64) where {TT,UU <: Any}

    any(t .== nL) ? sL = sum(La[t .== nL]) : sL = zero(TT)
    any(t .== nR) ? sR = sum(Ra[t .== nR]) : sR = zero(TT)

    σ2 = σ2_s * (sL + sR);   μ = -sL + sR + scaled_a_0

    transition_M!(F,σ2+σ2_a*dt,λ, μ, dx, xc, n, dt)
    P = F * P

    return P, F

end


"""
    latent_one_step!(P, F, λ, σ2_a, σ2_s, t, nL, nR, La, Ra, M, dx, xc, n, dt)
    for when bound is nonstationary

"""
function latent_one_step!(P::Vector{TT}, F::Array{TT,2}, λ::TT, σ2_a::TT, σ2_s::TT,
        t::Int, nL::Vector{Int}, nR::Vector{Int},
        La::Vector{TT}, Ra::Vector{TT}, scaled_a_0::TT,
        dx::Float64, xc::Vector{TT}, xc_pre::Vector{TT}, n::Int, dt::Float64) where {TT,UU <: Any}

    any(t .== nL) ? sL = sum(La[t .== nL]) : sL = zero(TT)
    any(t .== nR) ? sR = sum(Ra[t .== nR]) : sR = zero(TT)

    σ2 = σ2_s * (sL + sR);   μ = -sL + sR + scaled_a_0

    if size(F,1) == size(F,2)
        transition_M!(F,σ2+σ2_a*dt,λ, μ, dx, xc, n, dt)
    else
        transition_M!(F,σ2+σ2_a*dt,λ, μ, dx, xc, xc_pre, n, dt)
    end        
    P = F * P

    return P, F

end


"""
    bins(B,n)

Computes the bin center locations and bin spacing, given the boundary and number of bins.
"""
# function bins(B::TT, n::Int) where {TT}

#     dx = 2. *B/(n-2)

#     xc = vcat(collect(range(-(B+dx/2.),stop=-dx,length=Int((n-1)/2.))),0.,
#         collect(range(dx,stop=(B+dx/2.),length=Int((n-1)/2))))

#     return xc, dx

# end

function bins(B::TT, dx::Float64) where {TT <: Any}

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

function bins(B::TT, xc::Vector{TT}) where {TT <: Any}

    xc = xc[2:end-1]
    xc = vcat(xc, 2*B - xc[end])
    xc = vcat(-xc[end], xc)
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
julia> dt, n, B, σ2, λ, μ = 0.1, 53, 10., 10., -0.5, 1.;

julia> xc,dx = pulse_input_DDM.bins(B, n);

julia> M = pulse_input_DDM.transition_M(σ2, λ, μ, dx, xc, n, dt);

julia> size(M)
(53, 53)
```
"""
function transition_M(σ2::TT, λ::TT, μ::TT, dx::Float64,
        xc::Vector{TT}, n::Int, dt::Float64) where {TT<: Any}

    M = zeros(TT,n,n)
    transition_M!(M,σ2,λ,μ,dx,xc,n,dt)

    return M

end


"""
    transition_M!(F::Array{TT,2}, σ2::TT, λ::TT, μ::TT, dx::Float64,
        xc::Vector{TT}, n::Int, dt::Float64) where {TT <: Any}

"""
function transition_M!(F::Array{TT,2}, σ2::TT, λ::TT, μ::TT, dx::Float64,
        xc::Vector{TT}, n::Int, dt::Float64) where {TT,UU <: Any}

    F[1,1] = one(TT)
    F[n,n] = one(TT) 
    F[:,2:n-1] = zeros(TT,n,n-2)

    ndeltas = max(70,ceil(Int, 10. *sqrt(σ2)/dx))

    deltaidx = collect(-ndeltas:ndeltas)
    deltas = deltaidx * (5. *sqrt(σ2))/ndeltas
    ps = exp.(-0.5 * (5*deltaidx./ndeltas).^2)
    ps = ps/sum(ps)

    @inbounds for j = 2:n-1

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
    transition_M!(F::Array{TT,2}, σ2::TT, λ::TT, μ::TT, dx::Float64,
        xc::Vector{TT}, n::Int, dt::Float64) where {TT <: Any}

"""
function transition_M!(F, σ2::TT, λ::TT, μ::TT, dx::Float64,
        xc::Vector{TT}, xc_pre::Vector{TT}, n::Int, dt::Float64) where {TT,UU <: Any}

    n_pre = size(F,2)
    F[1,1] = one(TT); F[n,n_pre] = one(TT); # F[:,2:end-1] = zeros(TT,n,n_pre-2)

    ndeltas = max(70,ceil(Int, 10. *sqrt(σ2)/dx))

    deltaidx = collect(-ndeltas:ndeltas)
    deltas = deltaidx * (5. *sqrt(σ2))/ndeltas
    ps = exp.(-0.5 * (5*deltaidx./ndeltas).^2)
    ps = ps/sum(ps)

    @inbounds for j = 2:n_pre-1

        mu = exp(λ*dt)*xc_pre[j] + μ * expm1_div_x(λ*dt)

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
    make_adapted_clicks(ϕ, τ_ϕ, L, R)
    log-posterior space

"""
function adapt_clicks(ϕ::TT, τ_ϕ::TT, L::Vector{Float64}, R::Vector{Float64}, C::TT) where {TT}

    La, Ra = C.*ones(TT,length(L)), C.*ones(TT,length(R))
    return La, Ra

end



"""
    make_adapted_clicks(ϕ, τ_ϕ, L, R)
    Clicks space

"""
function adapt_clicks(ϕ::TT, τ_ϕ::TT, L::Vector{Float64}, R::Vector{Float64}, C::String) where {TT}


    if (length(L) > 1) & (length(R) > 1)

        if L[1] == R[1]
        # stereo click
            all = vcat(hcat(L[2:end], -1 * ones(length(L)-1)), hcat(R, ones(length(R))))
            all = all[sortperm(all[:, 1]), :]
            adapted = ones(TT, size(all,1))

            if (typeof(ϕ) == Float64) && (isapprox(ϕ, 1.0))
            else
                (length(all) > 1 && ϕ != 1.) ? adapt_clicks!(adapted, all[:, 1], ϕ, τ_ϕ) : nothing
            end

            all = vcat([L[1], -1.]', all)
            adapted = vcat(1., adapted)
            La, Ra = adapted[all[:,2] .== -1.], adapted[all[:,2] .== 1.]
        
        else
        
        # no stereo click
            all = vcat(hcat(L, -1 * ones(length(L))), hcat(R, ones(length(R))))
            all = all[sortperm(all[:, 1]), :]
            adapted = ones(TT, size(all,1))

            if (typeof(ϕ) == Float64) && (isapprox(ϕ, 1.0))
            else
                (length(all) > 1 && ϕ != 1.) ? adapt_clicks!(adapted, all[:, 1], ϕ, τ_ϕ) : nothing
            end

            La, Ra = adapted[all[:,2] .== -1.], adapted[all[:,2] .== 1.]

        end

    else

        La, Ra = ones(TT,length(L)), ones(TT,length(R))
        if (typeof(ϕ) == Float64) && (isapprox(ϕ, 1.0))
        else
            (length(L) > 1 && ϕ != 1.) ? adapt_clicks!(La, L, ϕ, τ_ϕ) : nothing
            (length(R) > 1 && ϕ != 1.) ? adapt_clicks!(Ra, R, ϕ, τ_ϕ) : nothing
        end
    end

    return La, Ra

end



"""
    adapt_clicks!(Ca, C, ϕ, τ_ϕ)

"""
function adapt_clicks!(Ca::Vector{TT}, C::Vector{Float64}, ϕ::TT, τ_ϕ::TT) where {TT}

    ici = diff(C)

    for i = 1:length(ici)
        arg = (1/τ_ϕ) * (-ici[i] + xlogy(τ_ϕ, abs(1. - Ca[i]*ϕ)))

        if Ca[i]*ϕ <= 1
            Ca[i+1] = 1. - exp(arg) 
        else
            Ca[i+1] = 1. + exp(arg)
        end
    end

end


"""
    make_adapted_clicks(ϕ, L, R) - just scaling the clicks with no temporal dynamics

"""
function adapt_clicks(ϕ::TT, L::Vector{Float64}, R::Vector{Float64}) where {TT}

    La, Ra = ϕ.*ones(TT,length(L)), ϕ.*ones(TT,length(R))

    return La, Ra

end

