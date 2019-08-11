
function LL_all_trials_new(pz::Vector{TT}, pd::Vector{TT}, data::Dict; n::Int=53) where {TT}
        
    dt = data["dt"]
    P,M,xc,dx, = initialize_latent_model(pz, n, dt)
            
    output = pmap((L,R,nT,nL,nR,choice) -> LL_single_trial(pz, pd, P, M, dx, xc,
        L, R, nT, nL, nR, choice, n, dt),
        data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"], 
        data["binned_rightbups"], data["pokedR"])   
    
end

function LL_single_trial(pz::Vector{TT}, pd::Vector{TT},
        P::Vector{TT}, M::Array{TT,2}, dx::TT, 
        xc::Vector{TT}, L::Vector{Float64}, R::Vector{Float64}, nT::Int,
        nL::Vector{Int}, nR::Vector{Int},
        pokedR::Bool, n::Int, dt::Float64) where {TT}
    
    #adapt magnitude of the click inputs
    La, Ra = make_adapted_clicks(pz,L,R)
            
    F = zeros(TT,n,n)
    
    @inbounds for t = 1:nT
        
        P,F = latent_one_step!(P,F,pz,t,nL,nR,La,Ra,M,dx,xc,n,dt)               
        (t == nT) && (P .*=  vcat(map(x-> exp(Bern_LL(Bern_sig(pd,x),pokedR)), xc)...))

    end

    return log(sum(P))

end

Bern_LL(p,x) = log(p)*x + log(1-p)*(1. -x)
Bern_sig(p,x) = p[1] + (p[2] - p[1]) * logistic((-p[3] * x) - p[4])