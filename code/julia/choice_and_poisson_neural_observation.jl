module choice_and_poisson_neural_observation

const dimz = 8

using global_functions, Optim, LineSearches
using poisson_neural_observation: inv_map_pz!, map_pz!, gauss_prior, poiss_LL, P_M_xc, opt_ll, transition_Pa!

function do_optim(pz,bias,py,fit_vec,map_str,dt,data,beta,mu0,f_str;n::Int=103,
    x_tol::Float64=1e-16,f_tol::Float64=1e-16,g_tol::Float64=1e-12)
    
    ###########################################################################################
    ## Map parameters to unbounded domain for optimization
    inv_map_pz!(pz,map_str,dt)     
    inv_map_py!.(py,f_str)

    ###########################################################################################
    ## Concatenate into a single vector and break up into optimization variables and constants
    p = vcat(pz,bias,vcat(py...))
    p_opt = p[fit_vec]
    p_const = p[.!fit_vec]

    ###########################################################################################
    ## Optimize
    ll(x) = ll_wrapper(x, p_const, fit_vec, data, f_str, beta=beta, mu0=mu0, n=n, dt=dt)
    p_opt = opt_ll(p_opt,ll;g_tol=g_tol,x_tol=x_tol,f_tol=f_tol);
    
    ###########################################################################################
    ## Break up optimization vector into functional groups and remap to bounded domain
    pz,py,bias = breakup(p_opt, p_const, fit_vec, f_str)
    map_pz!(pz,map_str,dt)       
    map_py!.(py,f_str)
    
    return pz, py, bias
    
end

function ll_wrapper{TT}(p_opt::Vector{TT}, p_const::Vector{Float64}, fit_vec::BitArray{1}, 
        data::Dict, f_str::String; map_str::String="exp",n::Int=203,
        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(0), 
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}(0),dt::Float64=2e-2)
        
    pz,py,bias = breakup(p_opt, p_const, fit_vec, f_str)
    map_pz!(pz,map_str,dt)       
    map_py!.(py,f_str)

    -(sum(LL_all_trials(pz, py, bias, data, f_str, n=n, dt=dt)) - sum(gauss_prior.(py,mu0,beta)))
              
end

function LL_all_trials{TT}(pz::Vector{TT},py::Union{Vector{Vector{TT}},Vector{Vector{Float64}}},bias::TT,
        data::Dict, f_str::String; comp_posterior::Bool=false, n::Int=203, dt::Float64=2e-2)
        
    P,M,xc,dx,xe = P_M_xc(pz,n=n,dt=dt)

    nbinsL, Sfrac = bias_bin(bias,xe,dx,n)
    lambday = fy.(xc,py',f_str)
            
    output = pmap((L,R,T,nL,nR,N,SC,choice) -> LL_single_trial(pz, P, M, dx, xc,
        L, R, T, nL, nR, lambday[:,N], SC, nbinsL, Sfrac, choice,
            comp_posterior=comp_posterior,n=n, dt=dt),
        data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"], 
        data["binned_rightbups"],data["N"],data["spike_counts"],data["pokedR"])   
    
end

function LL_single_trial{TT}(pz::Vector{TT}, P::Vector{TT}, M::Array{TT,2}, dx::TT, 
        xc::Vector{TT},L::Vector{Float64}, R::Vector{Float64}, T::Int,
        hereL::Vector{Int}, hereR::Vector{Int},
        lambday::Array{TT,2},spike_counts::Vector{Vector{Int}},
        nbinsL::Union{TT,Int}, Sfrac::TT, pokedR::Bool;
        comp_posterior::Bool=false, n::Int=203, dt::Float64=2e-2)
    
    #break up parameters
    phi,tau_phi = pz[7:8]
    #adapt magnitude of the click inputs
    La, Ra = make_adapted_clicks(phi,tau_phi,L,R)

    #vector to sum choice evidence
    pokedL = convert(TT,!pokedR); pokedR = convert(TT,pokedR)
    Pd = vcat(pokedL * ones(nbinsL), pokedL * Sfrac + pokedR * (one(Sfrac) - Sfrac), pokedR * ones(n - (nbinsL + 1)))
    
    #spike count data
    spike_counts = reshape(vcat(spike_counts...),:,length(spike_counts));          
    
    c = Vector{TT}(T)
    comp_posterior ? post = Array{Float64,2}(n,T) : nothing
    F = zeros(M)    #empty transition matrix for time bins with clicks

    @inbounds for t = 1:T
        
        P,F = transition_Pa!(P,F,pz,t,hereL,hereR,La,Ra,M,dx,xc,n,dt)               
        P .*= vec(exp.(sum(poiss_LL.(spike_counts[t,:],lambday',dt),1)));       
        (t == T) && (P .*=  Pd)
        c[t] = sum(P)
        P /= c[t] 
        comp_posterior ? post[:,t] = P : nothing

    end

    if comp_posterior

        P = ones(Float64,n); #initialze backward pass with all 1's
        post[:,T] .*= P;

        @inbounds for t = T-1:-1:1
            
            P .*= vec(exp.(sum(poiss_LL.(spike_counts[t+1,:],lambday',dt),1)));
            (t + 1 == T) && (P .*=  Pd)
            P,F = transition_Pa!(P,F,pz,t+1,hereL,hereR,La,Ra,M,dx,xc,n,dt;backwards=true)
            P /= c[t+1] 
            post[:,t] .*= P

        end

    end

    comp_posterior ? (return post) : (return sum(log.(c)))

end

function breakup{TT}(p_opt::Vector{TT}, p_const::Vector{Float64}, fit_vec::BitArray{1}, f_str::String)
    
    p = Vector{TT}(length(fit_vec))
    p[fit_vec] = p_opt;
    p[.!fit_vec] = p_const;
                
    pz = p[1:dimz];
    bias = p[9];

    if f_str == "sig"
        py = reshape(p[dimz+2:end],4,:)

    elseif f_str == "exp"
        py = reshape(p[dimz+2:end],2,:)

    elseif f_str == "softplus"
        py = reshape(p[dimz+2:end],3,:)

    end

    py = map(i->py[:,i],1:size(py,2))

    return pz, py, bias
    
end

end
