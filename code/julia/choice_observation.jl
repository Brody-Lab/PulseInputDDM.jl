module choice_observation

const dimz = 7

using latent_DDM_common_functions, ForwardDiff, Optim

function do_LL(p,dt,data,n)
    
    ###########################################################################################
    ## break up parameters based on which variables they relate to
    pz,bias = breakup(p)

    ###########################################################################################
    ## compute LL
    sum(LL_all_trials(pz, bias, data, n, dt))
    
end

function do_H(p,fit_vec,dt,data,n;map_str::String="exp")
    
    ###########################################################################################
    ## break up parameters based on which variables they relate to
    pz,bias = breakup(p)
    
    ###########################################################################################
    ## Map parameters to unbounded domain for optimization
    inv_map_pz!(pz,dt,map_str=map_str)     

    ###########################################################################################
    ## Concatenate into a single vector and break up into optimization variables and constants
    p_opt,p_const = inv_gather(inv_breakup(pz,bias),fit_vec)
    
    ###########################################################################################
    ## Break up optimization vector into functional groups and remap to bounded domain (to that
    ## function returns parameters in the bounded domain
    pz,bias = breakup(gather(p_opt, p_const, fit_vec))
    map_pz!(pz,dt,map_str=map_str)  

    ###########################################################################################
    ## Compute Hessian
    ll(x) = ll_wrapper(x, p_const, fit_vec, data, n, dt, map_str=map_str)

    H = ForwardDiff.hessian(ll, p_opt);
    d,V = eig(H)
    
    if all(d .> 0)
    
        CI = 2*sqrt.(diag(inv(H)));
    
        CIz_plus, CIbias_plus = breakup(gather(p_opt + CI, p_const, fit_vec))
        map_pz!(CIz_plus,dt,map_str=map_str) 

        CIz_minus, CIbias_minus = breakup(gather(p_opt - CI, p_const, fit_vec))
        map_pz!(CIz_minus,dt,map_str=map_str)
        
    else
        
        CIz_plus, CIz_minus = similar(pz),similar(pz)
        CIbias_plus, CIbias_minus = 1e-150,1e-150
        
    end
    
    CIplus = inv_breakup(CIz_plus,CIbias_plus) 
    CIminus = inv_breakup(CIz_minus,CIbias_minus)    
    
    return CIplus, CIminus, H
    
end

function do_optim(p,fit_vec,dt,data,n;map_str::String="exp",
    x_tol::Float64=1e-16,f_tol::Float64=1e-16,g_tol::Float64=1e-12,iterations::Int=Int(5e3))
    
    ###########################################################################################
    ## break up parameters based on which variables they relate to
    pz,bias = breakup(p)
    
    ###########################################################################################
    ## Map parameters to unbounded domain for optimization
    inv_map_pz!(pz,dt,map_str=map_str)     

    ###########################################################################################
    ## Concatenate into a single vector and break up into optimization variables and constants
    p_opt,p_const = inv_gather(inv_breakup(pz,bias),fit_vec)

    ###########################################################################################
    ## Optimize
    ll(x) = ll_wrapper(x, p_const, fit_vec, data, n, dt, map_str=map_str)
    opt_output, state = opt_ll(p_opt,ll;g_tol=g_tol,x_tol=x_tol,f_tol=f_tol,iterations=iterations);
    p_opt = Optim.minimizer(opt_output)

    ###########################################################################################
    ## Break up optimization vector into functional groups, remap to bounded domain and regroup
    pz,bias = breakup(gather(p_opt, p_const, fit_vec))
    p = inv_breakup(map_pz!(pz,dt,map_str=map_str),bias)  
    
    return p, opt_output, state
        
end

function ll_wrapper{TT}(p_opt::Vector{TT}, p_const::Vector{Float64}, fit_vec::Union{BitArray{1},Vector{Bool}}, 
        data::Dict, n::Int, dt::Float64; map_str::String="exp")
        
    pz,bias = breakup(gather(p_opt, p_const, fit_vec))
    map_pz!(pz,dt,map_str=map_str)    
    
    #lapse is like adding a prior out here?
    #modified 11/8 lapse needs to be dealt with different not clear that old way was correct
    #P *= (1.-inatt)
    #P += inatt/n

    -(sum(LL_all_trials(pz, bias, data, n, dt)))
              
end

function LL_all_trials{TT}(pz::Vector{TT},bias::TT,
        data::Dict, n::Int, dt::Float64; comp_posterior::Bool=false)
        
    P,M,xc,dx,xe = P_M_xc(pz,n,dt)

    nbinsL, Sfrac = bias_bin(bias,xe,dx,n)
            
    output = pmap((L,R,T,nL,nR,choice) -> LL_single_trial(pz, P, M, dx, xc,
        L, R, T, nL, nR, nbinsL, Sfrac, choice, n, dt, comp_posterior=comp_posterior),
        data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"], 
        data["binned_rightbups"],data["pokedR"])   
    
end

function LL_single_trial{TT}(pz::Vector{TT}, P::Vector{TT}, M::Array{TT,2}, dx::TT, 
        xc::Vector{TT},L::Vector{Float64}, R::Vector{Float64}, T::Int,
        hereL::Vector{Int}, hereR::Vector{Int},
        nbinsL::Union{TT,Int}, Sfrac::TT, pokedR::Bool,
        n::Int, dt::Float64;
        comp_posterior::Bool=false)
    
    #adapt magnitude of the click inputs
    La, Ra = make_adapted_clicks(pz,L,R)

    #vector to sum choice evidence
    pokedL = convert(TT,!pokedR); pokedR = convert(TT,pokedR)
    Pd = vcat(pokedL * ones(nbinsL), pokedL * Sfrac + pokedR * (one(Sfrac) - Sfrac), pokedR * ones(n - (nbinsL + 1)))
        
    c = Vector{TT}(T)
    comp_posterior ? post = Array{Float64,2}(n,T) : nothing
    F = zeros(M)     #empty transition matrix for time bins with clicks

    @inbounds for t = 1:T
        
        P,F = transition_Pa!(P,F,pz,t,hereL,hereR,La,Ra,M,dx,xc,n,dt)               
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
            P,F = transition_Pa!(P,F,pz,t+1,hereL,hereR,La,Ra,M,dx,xc,n,dt;backwards=true)
            P /= c[t+1] 
            post[:,t] .*= P

        end

    end

    comp_posterior ? (return post) : (return sum(log.(c)))

end

breakup{TT}(p::Vector{TT}) = p[1:dimz],p[dimz+1]
inv_breakup{TT}(pz::Vector{TT},bias::TT) = cat(1,pz,bias)
    
function sampled_dataset!(data::Dict, p::Vector{Float64}; dtMC::Float64=1e-4, num_reps::Int=1, rng::Int = 1)
        
    construct_inputs!(data,num_reps)
    
    srand(rng)
    data["pokedR"] = pmap((T,leftbups,rightbups,rng) -> sample_model(T,leftbups,rightbups,p,rng=rng),
        data["T"],data["leftbups"],data["rightbups"],shuffle(1:length(data["N"])));
            
    return data
    
end

function sample_model(T::Float64,L::Vector{Float64},R::Vector{Float64},
        p::Vector{Float64};dtMC::Float64=1e-4,rng::Int=1)
    
    srand(rng)
    pz,bias = breakup(p)
    A = sample_latent(T,L,R,pz;dt=dtMC)
            
    choice = A[end] >= bias;
    
end

function bias_bin{TT}(bias::TT,xe::Vector{TT},dx::TT,n::Int)
    
    #nbinsL = ceil(Int,(B+bias)/dx)
    #Sfrac = one(dx)/dx * (bias - (-(B+dx)+nbinsL*dx))
    nbinsL = sum(bias .> xe[2:n])
    Sfrac = (bias - xe[nbinsL+1])/dx
    Sfrac < zero(Sfrac) ? Sfrac = zero(Sfrac) : nothing
    Sfrac > one(Sfrac) ? Sfrac = one(Sfrac) : nothing
    
    return nbinsL, Sfrac
    
end

end
