
const dimz = 7

function initialize_latent_model(pz::Vector{TT}, dx::Float64, dt::Float64;
        L_lapse::UU=0., R_lapse::UU=0.) where {TT,UU}

    σ2_i,B,λ,σ2_a = pz[1:4]                      #break up latent variables

    xc,n = bins(B,dx)                              # spatial bin centers, width and edges

    P = P0(σ2_i,n,dx,xc,dt;
        L_lapse=L_lapse, R_lapse=R_lapse)             # make initial latent distribution

    M = zeros(TT,n,n)                                 # build empty transition matrix
    M!(M,σ2_a*dt,λ,zero(TT),dx,xc,n,dt)          # build state transition matrix for no input time bins

    return P, M, xc, n

end

function P0(σ2_i::TT, n::Int, dx::VV, xc::Vector{WW}, dt::Float64;
        L_lapse::UU=0., R_lapse::UU=0.) where {TT,UU,VV,WW <: Any}

    P = zeros(TT,n)
    P[ceil(Int,n/2)] = one(TT) - (L_lapse + R_lapse)     # make initial delta function
    P[1], P[n] = L_lapse, R_lapse
    M = zeros(WW,n,n)                                    # build empty transition matrix
    M!(M,σ2_i,zero(WW),zero(WW),dx,xc,n,dt)
    P = M * P

end

function latent_one_step!(P::Vector{TT}, F::Array{TT,2}, pz::Vector{WW}, t::Int,
        nL::Vector{Int}, nR::Vector{Int},
        La::Vector{YY}, Ra::Vector{YY}, M::Array{TT,2},
        dx::UU, xc::Vector{VV}, n::Int, dt::Float64; backwards::Bool=false) where {TT,UU,VV,WW,YY <: Any}

    λ, σ2_a, σ2_s = pz[3:5]

    any(t .== nL) ? sL = sum(La[t .== nL]) : sL = zero(TT)
    any(t .== nR) ? sR = sum(Ra[t .== nR]) : sR = zero(TT)

    σ2 = σ2_s * (sL + sR);   μ = -sL + sR

    if backwards
        (sL + sR) > zero(TT) ? (M!(F,σ2+σ2_a*dt,λ, μ/dt, dx, xc, n, dt); P  = F' * P;) : P = M' * P
    else
        (sL + sR) > zero(TT) ? (M!(F,σ2+σ2_a*dt,λ, μ/dt, dx, xc, n, dt); P  = F * P;) : P = M * P
    end

    return P, F

end

function bins(B::TT,dx::Float64) where {TT}

    xc = collect(0.:dx:value(B))

    if xc[end] == B
        xc = vcat(xc[1:end-1], B + dx)
    else
        xc = vcat(xc, 2*B - xc[end])
    end

    xc = vcat(-xc[end:-1:2], xc)
    n = length(xc)

    return xc, n

end

function M!(F::Array{WW,2}, σ2::YY, λ::ZZ, h::Union{TT}, dx::UU, xc::Vector{VV}, n::Int, dt::Float64) where {TT,UU,VV,WW,YY,ZZ <: Any}

    F[1,1] = one(TT); F[n,n] = one(TT); F[:,2:n-1] = zeros(TT,n,n-2)

    #########################################

    ndeltas = max(70,ceil(Int, 10. *sqrt(σ2)/dx))

    deltaidx = collect(-ndeltas:ndeltas)
    deltas = deltaidx * (5. *sqrt(σ2))/ndeltas
    ps = exp.(-0.5 * (5*deltaidx./ndeltas).^2)
    ps = ps/sum(ps)

    @inbounds for j = 2:n-1

        abs(λ) < 1e-150 ? mu = xc[j] + h * dt : mu = exp(λ*dt)*(xc[j] + h/λ) - h/λ

        #now we're going to look over all the slices of the gaussian
        for k = 1:2*ndeltas+1

            s = mu + deltas[k]

            if s <= xc[1]

                F[1,j] += ps[k];

            elseif s >= xc[n]

                F[n,j] += ps[k];

            else

                if xc[1] < s && xc[2] > s

                    lp,hp = 1,2;

                elseif xc[n-1] < s && xc[n] > s

                    lp,hp = n-1,n;

                else

                    hp,lp = ceil(Int, (s-xc[2])/dx) + 2, floor(Int, (s-xc[2])/dx) + 2;

                end

                if (hp == lp)

                    F[lp,j] += ps[k];

                else

                    dd = xc[hp] - xc[lp];
                    F[hp,j] += ps[k]*(s-xc[lp])/dd;
                    F[lp,j] += ps[k]*(xc[hp]-s)/dd;

                end

            end

        end

    end

end

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

function adapt_clicks!(Ca::Vector{TT},  ϕ::TT, τ_ϕ::TT, ici::Vector{Float64}) where {TT}

    for i = 1:length(ici)
        arg = xlogy(τ_ϕ, abs(1. - Ca[i]* ϕ))
        Ca[i+1] = 1. - exp((-ici[i] + arg)/τ_ϕ)
    end

end
