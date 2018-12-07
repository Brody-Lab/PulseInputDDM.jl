module choice_and_poisson_neural_observation

using latent_DDM_common_functions, ForwardDiff, Optim, Pandas, Distributions
using poisson_neural_observation: gauss_prior, poiss_LL, inv_map_py!, map_py!, fy
using choice_observation: bias_bin
using Distributed, SpecialFunctions, LinearAlgebra

function do_optim(p,fit_vec,dt,data,n;
        f_str="softplus",map_str::String="exp",
        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        x_tol::Float64=1e-16,f_tol::Float64=1e-16,g_tol::Float64=1e-12,
        iterations::Int=Int(5e3),show_trace::Bool=false)  
    
    ###########################################################################################
    ## break up parameters based on which variables they relate to
    pz, py, bias = breakup(p,f_str=f_str)
    
    ###########################################################################################
    ## Map parameters to unbounded domain for optimization
    inv_map_pz!(pz,dt,map_str=map_str)     
    inv_map_py!.(py,f_str=f_str)

    ###########################################################################################
    ## Concatenate into a single vector and break up into optimization variables and constants
    p_opt,p_const = inv_gather(inv_breakup(pz,bias,py),fit_vec)

    ###########################################################################################
    ## Optimize
    ll(x) = ll_wrapper(x, p_const, fit_vec, data, dt, n; 
        f_str=f_str, beta=beta, mu0=mu0, map_str=map_str)
    opt_output, state = opt_ll(p_opt,ll;g_tol=g_tol,x_tol=x_tol,f_tol=f_tol,iterations=iterations,
        show_trace=show_trace);
    p_opt = Optim.minimizer(opt_output)
    
    ###########################################################################################
    ## Break up optimization vector into functional groups, remap to bounded domain and regroup
    pz,py,bias = breakup(gather(p_opt, p_const, fit_vec),f_str=f_str)
    map_pz!(pz,dt,map_str=map_str)       
    map_py!.(py,f_str=f_str)
    p = vcat(pz,bias,py)   
        
    return p, opt_output, state
    
end

function do_LL(p,dt,data,n::Int;
        f_str="softplus",map_str::String="exp",
        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}())
    
    ###########################################################################################
    ## break up parameters based on which variables they relate to
    pz, py, bias = breakup(p,f_str=f_str)

    ###########################################################################################
    ## Compute LL
    LL = sum(LL_all_trials(pz, py, bias, data, dt, n, f_str=f_str))
    
    length(beta) > 0 ? LL += sum(gauss_prior.(py,mu0,beta)) : nothing
    
    return LL
    
end

function ll_wrapper(p_opt::Vector{TT}, p_const::Vector{Float64}, fit_vec::Union{BitArray{1},Vector{Bool}}, 
        data::Dict, dt::Float64, n::Int; f_str::String="softplus", map_str::String="exp",
        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(0), 
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}(0)) where {TT}
           
    pz,py,bias = breakup(gather(p_opt, p_const, fit_vec),f_str=f_str)
    map_pz!(pz,dt,map_str=map_str)       
    map_py!.(py,f_str=f_str)

    LL = sum(LL_all_trials(pz, py, bias, data, dt, n, f_str=f_str))
    
    length(beta) > 0 ? LL += sum(gauss_prior.(py,mu0,beta)) : nothing    
    
    return -LL
              
end

function LL_all_trials(pz::Vector{TT},py::Union{Vector{Vector{TT}},Vector{Vector{Float64}}},bias::TT,
        data::Dict,dt::Float64,n::Int; f_str::String="softplus", comp_posterior::Bool=false) where {TT}
        
    P,M,xc,dx,xe = P_M_xc(pz,n,dt)

    nbinsL, Sfrac = bias_bin(bias,xe,dx,n)
    lambday = fy.(py,xc',f_str=f_str)'
            
    output = pmap((L,R,T,nL,nR,N,SC,choice) -> LL_single_trial(pz, P, M, dx, xc,
        L, R, T, nL, nR, lambday[:,N], SC, nbinsL, Sfrac, choice, dt, n;
            comp_posterior=comp_posterior),
        data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"], 
        data["binned_rightbups"],data["N"],data["spike_counts"],data["pokedR"])   
    
end

function LL_single_trial(pz::Vector{TT}, P::Vector{TT}, M::Array{TT,2}, dx::TT, 
        xc::Vector{TT},L::Vector{Float64}, R::Vector{Float64}, T::Int,
        hereL::Vector{Int}, hereR::Vector{Int},
        lambday::Array{TT,2},spike_counts::Vector{Vector{Int}},
        nbinsL::Union{TT,Int}, Sfrac::TT, pokedR::Bool, dt::Float64, n::Int;
        comp_posterior::Bool=false) where {TT}
    
    #adapt magnitude of the click inputs
    La, Ra = make_adapted_clicks(pz,L,R)

    #vector to sum choice evidence
    pokedL = convert(TT,!pokedR); pokedR = convert(TT,pokedR)
    Pd = vcat(pokedL * ones(nbinsL), pokedL * Sfrac + pokedR * (one(Sfrac) - Sfrac), pokedR * ones(n - (nbinsL + 1)))
    
    #spike count data
    spike_counts = reshape(vcat(spike_counts...),:,length(spike_counts));          
    
    c = Vector{TT}(undef,T)
    comp_posterior ? post = Array{Float64,2}(undef,n,T) : nothing
    F = zeros(TT,n,n)    #empty transition matrix for time bins with clicks

    @inbounds for t = 1:T
        
        P,F = transition_Pa!(P,F,pz,t,hereL,hereR,La,Ra,M,dx,xc,n,dt)               
        P .*= vec(exp.(sum(poiss_LL.(spike_counts[t,:],lambday',dt),dims=1)));       
        (t == T) && (P .*=  Pd)
        c[t] = sum(P)
        P /= c[t] 
        comp_posterior ? post[:,t] = P : nothing

    end

    if comp_posterior

        P = ones(Float64,n); #initialze backward pass with all 1's
        post[:,T] .*= P;

        @inbounds for t = T-1:-1:1
            
            P .*= vec(exp.(sum(poiss_LL.(spike_counts[t+1,:],lambday',dt),dims=1)));
            (t + 1 == T) && (P .*=  Pd)
            P,F = transition_Pa!(P,F,pz,t+1,hereL,hereR,La,Ra,M,dx,xc,n,dt;backwards=true)
            P /= c[t+1] 
            post[:,t] .*= P

        end

    end

    comp_posterior ? (return post) : (return sum(log.(c)))

end

function breakup(p::Vector{TT}; f_str::String="softplus") where {TT}
                
    pz = p[1:dimz];
    bias = p[dimz+1];

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

inv_breakup(pz::Vector{TT},bias::TT, py::Vector{Vector{TT}}) where {TT} = vcat(pz,bias,vcat(py...))

end
