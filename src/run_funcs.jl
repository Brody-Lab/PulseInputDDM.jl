
"""
    do_H

    returns Hessian

"""

function do_H(p,fit_vec,dt,data,n::Int;
        f_str="softplus",map_str::String="exp",
        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}())
    
    ###########################################################################################
    ## break up parameters based on which variables they relate to
    pz,py = breakup(p,f_str=f_str)
    
    ###########################################################################################
    ## Map parameters to unbounded domain for optimization
    inv_map_pz!(pz,dt,map_str=map_str)     
    inv_map_py!.(py,f_str=f_str)

    ###########################################################################################
    ## Concatenate into a single vector and break up into optimization variables and constants
    p_opt,p_const = inv_gather(inv_breakup(pz,py),fit_vec)
    
    ###########################################################################################
    ## Break up optimization vector into functional groups, remap to bounded domain and regroup
    pz,py = breakup(gather(p_opt, p_const, fit_vec),f_str=f_str)
    map_pz!(pz,dt,map_str=map_str)       
    map_py!.(py,f_str=f_str)

    ###########################################################################################
    ## compute Hessian
    ll(x) = ll_wrapper(x, p_const, fit_vec, data, dt, n; 
        f_str=f_str, beta=beta, mu0=mu0, map_str=map_str)
    
    H = ForwardDiff.hessian(ll, p_opt);
    d,V = eig(H)
    
    if all(d .> 0)
    
        CI = 2*sqrt.(diag(inv(H)));
    
        CIz_plus, CIpy_plus = breakup(gather(p_opt + CI, p_const, fit_vec),f_str=f_str)
        map_pz!(CIz_plus,dt,map_str=map_str)
        map_py!.(CIpy_plus,f_str=f_str)

        CIz_minus, CIpy_minus = breakup(gather(p_opt - CI, p_const, fit_vec),f_str=f_str)
        map_pz!(CIz_minus,dt,map_str=map_str)
        map_py!.(CIpy_minus,f_str=f_str)
    
    else
        
        CIz_plus, CIz_minus = similar(pz),similar(pz)
        CIpy_plus, CIpy_minus = map(x->similar(x),deepcopy(py)),map(x->similar(x),deepcopy(py))
        
    end
    
    CIplus = vcat(CIz_plus,CIpy_plus) 
    CIminus = vcat(CIz_minus,CIpy_minus)    
    
    return CIplus, CIminus, H
    
end

#=

function do_optim(pz,py,pRBF,fit_vec,dt,data,n::Int;
        f_str="softplus",map_str::String="exp",
        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        x_tol::Float64=1e-16,f_tol::Float64=1e-16,g_tol::Float64=1e-12,
        iterations::Int=Int(5e3),show_trace::Bool=false)
    
    ###########################################################################################
    ## Map parameters to unbounded domain for optimization
    inv_map_pz!(pz,dt,map_str=map_str)     
    inv_map_py!.(py,f_str=f_str)

    ###########################################################################################
    ## Concatenate into a single vector and break up into optimization variables and constants
    p_opt,p_const = inv_gather(inv_breakup(pz,py,pRBF),fit_vec)

    ###########################################################################################
    ## Optimize
    ll(x) = ll_wrapper_RBF(x, p_const, fit_vec, data, dt, n; 
        f_str=f_str, beta=beta, mu0=mu0, map_str=map_str)
    opt_output, state = opt_ll(p_opt,ll;g_tol=g_tol,x_tol=x_tol,f_tol=f_tol,iterations=iterations,
        show_trace=show_trace);
    p_opt = Optim.minimizer(opt_output)

    ###########################################################################################
    ## Break up optimization vector into functional groups, remap to bounded domain and regroup
    pz,py = breakup(gather(p_opt, p_const, fit_vec),f_str=f_str)
    map_pz!(pz,dt,map_str=map_str)       
    map_py!.(py,f_str=f_str)
        
    return pz, py, pRBF, opt_output, state
    
end

=#

function do_optim(p,fit_vec,dt,data,n::Int;
        f_str="softplus",map_str::String="exp",
        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        x_tol::Float64=1e-16,f_tol::Float64=1e-16,g_tol::Float64=1e-12,
        iterations::Int=Int(5e3),show_trace::Bool=false, muf::Vector{Vector{Float64}}=Vector{Vector{Float64}}())
    
    ###########################################################################################
    ## break up parameters based on which variables they relate to
    pz,py = breakup(p,f_str=f_str)
    
    ###########################################################################################
    ## Map parameters to unbounded domain for optimization
    inv_map_pz!(pz,dt,map_str=map_str)     
    inv_map_py!.(py,f_str=f_str)

    ###########################################################################################
    ## Concatenate into a single vector and break up into optimization variables and constants
    p_opt,p_const = inv_gather(inv_breakup(pz,py),fit_vec)

    ###########################################################################################
    ## Optimize
    ll(x) = ll_wrapper(x, p_const, fit_vec, data, dt, n; 
        f_str=f_str, beta=beta, mu0=mu0, map_str=map_str,muf=muf)
    opt_output, state = opt_ll(p_opt,ll;g_tol=g_tol,x_tol=x_tol,f_tol=f_tol,iterations=iterations,
        show_trace=show_trace);
    p_opt = Optim.minimizer(opt_output)

    ###########################################################################################
    ## Break up optimization vector into functional groups, remap to bounded domain and regroup
    pz,py = breakup(gather(p_opt, p_const, fit_vec),f_str=f_str)
    map_pz!(pz,dt,map_str=map_str)       
    map_py!.(py,f_str=f_str)
    p = vcat(pz,py)   
        
    return p, opt_output, state
    
end

#function do_LL(p,fit_vec,dt,data,n::Int;
#        f_str="softplus",map_str::String="exp",
#        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
#        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}())

function do_LL(p,dt,data,n::Int;
        f_str="softplus",map_str::String="exp",
        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}())
    
    ###########################################################################################
    ## break up parameters based on which variables they relate to
    pz,py = breakup(p,f_str=f_str)
    
    ###########################################################################################
    ## Map parameters to unbounded domain for optimization
    #inv_map_pz!(pz,dt,map_str=map_str)     
    #inv_map_py!.(py,f_str=f_str)

    ###########################################################################################
    ## Concatenate into a single vector and break up into optimization variables and constants
    #p_opt,p_const = inv_gather(inv_breakup(pz,py),fit_vec)
    
    ###########################################################################################
    ## Break up optimization vector into functional groups, remap to bounded domain and regroup
    #pz,py = breakup(gather(p_opt, p_const, fit_vec),f_str=f_str)
    #map_pz!(pz,dt,map_str=map_str)       
    #map_py!.(py,f_str=f_str)

    ###########################################################################################
    ## Compute LL
    #ll_wrapper(p_opt, p_const, fit_vec, data, dt, n; f_str=f_str, beta=beta, mu0=mu0, map_str=map_str)
    
    #11/5 changed this to deal with no priors, which is a default now
    #-(sum(LL_all_trials(pz, py, data, f_str=f_str, n=n, dt=dt)) - sum(gauss_prior.(py,mu0,beta)))
    LL = sum(LL_all_trials(pz, py, data, dt, n, f_str=f_str))
    
    length(beta) > 0 ? LL += sum(gauss_prior.(py,mu0,beta)) : nothing
    
    return LL
    
end

function my_qcut(y,nconds)

    qvec = nquantile(y,nconds)
    qidx = map(x->findfirst(x .<= qvec),y)
    qidx[qidx .== 1] .= 2
    qidx .= qidx .- 2

end

function compute_p0(ΔLR,k,dt;f_str::String="softplus",nconds::Int=7);
    
    #### compute linear regression slope of tuning to $\Delta_{LR}$ and miniumum firing based on binning and averaging

    #conds_bins = my_qcut(vcat(ΔLR...),nconds)
    conds_bins, = qcut(vcat(ΔLR...),nconds,labels=false,duplicates="drop",retbins=true)
    fr = map(i -> (1/dt)*mean(vcat(k...)[conds_bins .== i]),0:nconds-1)

    #c = linreg(vcat(ΔLR...),vcat(k...))
    A = vcat(ΔLR...)
    b = vcat(k...)
    c = hcat(ones(size(A, 1)), A) \ b

    if f_str == "exp"
        p = vcat(minimum(fr),c[2])
    elseif f_str == "sig"
        p = vcat(minimum(fr),maximum(fr)-minimum(fr),c[2],0.)
    elseif f_str == "sig2"
        p = vcat(minimum(fr),maximum(fr)-minimum(fr),c[2],0.)
    elseif f_str == "softplus"
        p = vcat(minimum(fr),c[2],0.)
    end
        
end

function do_p0(dt::Float64,data::Dict;f_str::String="softplus")
    
    ###########################################################################################
    ## Compute click difference and organize spikes by neuron
    ΔLR = pmap((T,L,R)->diffLR(T,L,R,data["dt"]),data["nT"],data["leftbups"],data["rightbups"])    
    trials, SC = group_by_neuron(data)
    
    pmap((trials,k)->compute_p0(ΔLR[trials],k,dt;f_str=f_str),trials,SC)
    
end
    
function do_optim_ΔLR(dt::Float64,data::Dict,fit_vec::Union{Vector{BitArray{1}},Vector{Vector{Bool}}};
        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        f_str::String="softplus",
        x_tol::Float64=1e-16,f_tol::Float64=1e-16,g_tol::Float64=1e-12,
        iterations::Int=Int(5e3),show_trace::Bool=false,
        muf::Vector{Vector{Float64}}=Vector{Vector{Float64}}())
    
        ###########################################################################################
        ## Compute p0 by linear regression
        p = do_p0(dt,data;f_str=f_str) 

        ###########################################################################################
        ## Compute click difference and organize spikes by neuron
        ΔLR = pmap((T,L,R)->diffLR(T,L,R,data["dt"]),data["nT"],data["leftbups"],data["rightbups"])    
        trials, SC = group_by_neuron(data)   
    
        ###########################################################################################
        ## Map parameters to unbounded domain for optimization
        inv_map_py!.(p,f_str=f_str)
    
        p = pmap((p,trials,k,fit_vec,muf)->do_optim_ΔLR_single(p,dt,ΔLR[trials],k,fit_vec;
            show_trace=show_trace,f_str=f_str,muf=muf),p,trials,SC,fit_vec,muf)
    
        ###########################################################################################
        ## Remap to bounded domain
        map_py!.(p,f_str=f_str)
    
end

function do_optim_ΔLR_single(p::Vector{Float64},dt::Float64,ΔLR::Vector{Vector{Int}},
        k::Vector{Vector{Int}},fit_vec::Union{BitArray{1},Vector{Bool}};
        beta::Vector{Float64}=Vector{Float64}(),
        mu0::Vector{Float64}=Vector{Float64}(),f_str::String="softplus",
        x_tol::Float64=1e-16,f_tol::Float64=1e-16,g_tol::Float64=1e-12,iterations::Int=Int(5e3),
        show_trace::Bool=false,
        muf::Vector{Float64}=Vector{Float64}())
        
        p_opt,p_const = inv_gather(p,fit_vec)
    
        ###########################################################################################
        ## Optimize    
        ll(p_opt) = ll_wrapper_ΔLR(p_opt, p_const, fit_vec, k, ΔLR, dt; beta=beta, mu0=mu0, f_str=f_str, muf=muf)
        opt_output, state = opt_ll(p_opt,ll;g_tol=g_tol,x_tol=x_tol,f_tol=f_tol,iterations=iterations,show_trace=show_trace)
        p_opt = Optim.minimizer(opt_output)
    
        p = gather(p_opt, p_const, fit_vec)
            
        return p
                
end

function ll_wrapper_ΔLR(p_opt::Vector{TT}, p_const::Vector{Float64},fit_vec::Union{BitArray{1},Vector{Bool}},
        k::Vector{Vector{Int}}, ΔLR::Vector{Vector{Int}}, dt::Float64;
        beta::Vector{Float64}=Vector{Float64}(0),
        mu0::Vector{Float64}=Vector{Float64}(0),
        f_str::String="softplus",
        muf::Vector{Float64}=Vector{Float64}()) where {TT}
    
        p = gather(p_opt, p_const, fit_vec)
       
        #check fy because of NaN poiss_LL fiasco
        map_py!(p,f_str=f_str)
        λ = fy.([p],vcat(ΔLR...),f_str=f_str)
        λ0 = vcat(map(x->muf[1:length(x)],ΔLR)...)
    
        #-(sum(poiss_LL(λ,vcat(k...),dt)) - sum(gauss_prior(py,mu0,beta)))
        #LL = sum(poiss_LL.(vcat(k...),λ,dt))
        LL = sum(poiss_LL.(vcat(k...),λ+λ0,dt))
        length(beta) > 0 ? LL += sum(gauss_prior.(p,mu0,beta)) : nothing
    
        return -LL
            
end
