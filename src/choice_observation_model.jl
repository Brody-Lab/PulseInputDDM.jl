
function LL_all_trials(pz::Vector{TT},bias::TT,
        data::Dict; n::Int=53, dt::Float64=1e-2, 
        comp_posterior::Bool=false) where {TT}
        
    P,M,xc,dx,xe = initialize_latent_model(pz,n,dt)

    nbinsL, Sfrac = bias_bin(bias,xe,dx,n)
            
    output = pmap((L,R,T,nL,nR,choice) -> LL_single_trial(pz, P, M, dx, xc,
        L, R, T, nL, nR, nbinsL, Sfrac, choice, n, dt, comp_posterior=comp_posterior),
        data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"], 
        data["binned_rightbups"],data["pokedR"])   
    
end

function LL_single_trial(pz::Vector{TT}, P::Vector{TT}, M::Array{TT,2}, dx::TT, 
        xc::Vector{TT},L::Vector{Float64}, R::Vector{Float64}, T::Int,
        hereL::Vector{Int}, hereR::Vector{Int},
        nbinsL::Union{TT,Int}, Sfrac::TT, pokedR::Bool,
        n::Int, dt::Float64;
        comp_posterior::Bool=false) where {TT}
    
    #adapt magnitude of the click inputs
    La, Ra = make_adapted_clicks(pz,L,R)

    #vector to sum choice evidence
    pokedL = convert(TT,!pokedR); pokedR = convert(TT,pokedR)
    Pd = vcat(pokedL * ones(nbinsL), pokedL * Sfrac + pokedR * (one(Sfrac) - Sfrac), pokedR * ones(n - (nbinsL + 1)))
        
    c = Vector{TT}(undef,T)
    comp_posterior ? post = Array{Float64,2}(unde,n,T) : nothing
    F = zeros(TT,n,n)     #empty transition matrix for time bins with clicks

    @inbounds for t = 1:T
        
        P,F = latent_one_step!(P,F,pz,t,hereL,hereR,La,Ra,M,dx,xc,n,dt)               
        (t == T) && (P .*=  Pd)
        c[t] = sum(P)
        P /= c[t] 
        comp_posterior ? post[:,t] = P : nothing

    end

    if comp_posterior

        P = ones(Float64,n); #initialze backward pass with all 1's  
        post[:,T] .*= P;

        @inbounds for t = T-1:-1:1
            
            (t + 1 == T) && (P .*=  Pd)
            P,F = latent_one_step!(P,F,pz,t+1,hereL,hereR,La,Ra,M,dx,xc,n,dt;backwards=true)
            P /= c[t+1] 
            post[:,t] .*= P

        end

    end

    comp_posterior ? (return post) : (return sum(log.(c)))

end

function bias_bin(bias::TT,xe::Vector{TT},dx::TT,n::Int) where {TT}
    
    #nbinsL = ceil(Int,(B+bias)/dx)
    #Sfrac = one(dx)/dx * (bias - (-(B+dx)+nbinsL*dx))
    nbinsL = sum(bias .> xe[2:n])
    Sfrac = (bias - xe[nbinsL+1])/dx
    Sfrac < zero(Sfrac) ? Sfrac = zero(Sfrac) : nothing
    Sfrac > one(Sfrac) ? Sfrac = one(Sfrac) : nothing
    
    return nbinsL, Sfrac
    
end

choice_null(choices) = sum(choices .== true)*log(sum(choices .== true)/length(choices)) + 
    sum(choices .== false)*log(sum(choices .== false)/length(choices))