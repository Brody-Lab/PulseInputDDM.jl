
function LL_all_trials(pz::Vector{TT}, pw::Vector{TT}, pd::Vector{TT}, data::Dict; n::Int=53) where {TT}
        
    bias,lapse = pd[1],pd[2]
    dt = data["dt"]
    P,M,xc,dx,xe = initialize_latent_model(pz, n,dt, L_lapse=lapse/2, R_lapse=lapse/2)

    nbinsL, Sfrac = bias_bin(bias,xe,dx,n)
            
    output = pmap((L_l, R_l, nT, nL_l, nR_l, choice, L_f, R_f, nL_f, nR_f, context) -> LL_single_trial(pz, pw, P, M, dx, xc,
        L_l, R_l, L_f, R_f, nT, nL_l, nR_l, nL_f, nR_f, nbinsL, Sfrac, choice, n, dt, context),
        data["leftbups_loc"], data["rightbups_loc"], data["nT"], data["binned_leftbups_loc"], 
        data["binned_rightbups_loc"],data["pokedR"],
        data["leftbups_freq"], data["rightbups_freq"], data["binned_leftbups_freq"], 
        data["binned_rightbups_freq"],data["context_loc"])   
    
end

function LL_single_trial(pz::Vector{TT}, pw::Vector{TT},
        P::Vector{TT}, M::Array{TT,2}, dx::UU, 
        xc::Vector{TT},
        L_l::Vector{Float64}, R_l::Vector{Float64}, 
        L_f::Vector{Float64}, R_f::Vector{Float64}, 
        nT::Int,
        nL_l::Vector{Int}, nR_l::Vector{Int},
        nL_f::Vector{Int}, nR_f::Vector{Int},
        nbinsL::Union{TT,Int}, Sfrac::TT, pokedR::Bool,
        n::Int, dt::Float64, context::Bool) where {TT,UU <: Any}
    
    #adapt magnitude of the click inputs
    La_l, Ra_l = make_adapted_clicks(pz,L_l,R_l)
    La_f, Ra_f = make_adapted_clicks(pz,L_f,R_f)
    
    if context #location context is true...
        w_loc, w_freq = pw[1:2] #...set location weight to active
    else #if location context if false, i.e. freq context
        w_freq, w_loc = pw[1:2] #...set freq weight to active
    end

    #vector to sum choice evidence
    pokedL = convert(TT,!pokedR); pokedR = convert(TT,pokedR)
    Pd = vcat(pokedL * ones(nbinsL), pokedL * Sfrac + pokedR * (one(Sfrac) - Sfrac), pokedR * ones(n - (nbinsL + 1)))
        
    F = zeros(TT,n,n)     #empty transition matrix for time bins with clicks

    @inbounds for t = 1:nT
        
        P,F = latent_one_step!(P,F,pz,w_loc,w_freq,t,nL_l,nR_l,La_l,Ra_l,
            nL_f,nR_f,La_f,Ra_f,M,dx,xc,n,dt)          
        (t == nT) && (P .*=  Pd)

    end

    return log(sum(P))

end

function latent_one_step!(P::Vector{TT},F::Array{TT,2},pz::Vector{WW},
        w_loc::WW, w_freq::WW, t::Int,
        nL_l::Vector{Int}, nR_l::Vector{Int},
        La_l::Vector{YY}, Ra_l::Vector{YY}, 
        nL_f::Vector{Int}, nR_f::Vector{Int},
        La_f::Vector{YY}, Ra_f::Vector{YY}, 
        M::Array{TT,2},
        dx::UU,xc::Vector{VV},
        n::Int,dt::Float64) where {TT,UU,VV,WW,YY <: Any}
    
    lambda,vara,vars = pz[3:5]
    
    #any(t .== nL_l) ? sL_l = sum(La_l[t .== nL_l]) : sL_l = zero(TT)
    #any(t .== nR_l) ? sR_l = sum(Ra_l[t .== nR_l]) : sR_l = zero(TT)
    
    #any(t .== nL_f) ? sL_f = sum(La_f[t .== nL_f]) : sL_f = zero(TT)
    #any(t .== nR_f) ? sR_f = sum(Ra_f[t .== nR_f]) : sR_f = zero(TT)
    
    sum_clicks, mu = make_inputs(t, nL_l, nR_l, nL_f, nR_f, La_l, Ra_l, La_f, Ra_f, w_loc, w_freq)

    #sum_clicks = (w_loc * (sL_l + sR_l) + w_freq * (sL_f + sR_f))
    var = vars * sum_clicks
    #mu = w_loc * (-sL_l + sR_l) + w_freq * (-sL_f + sR_f)

    sum_clicks > zero(TT) ? (M!(F,var+vara*dt,lambda,mu/dt,dx,xc,n,dt); P  = F * P;) : P = M * P
    
    return P, F
    
end