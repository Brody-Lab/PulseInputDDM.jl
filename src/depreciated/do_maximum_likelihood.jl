module do_maximum_likelihood

using latent_model, LineSearches, Optim, global_functions, ForwardDiff
using sample_model, initialize_spike_obs_model, StatsBase, Pandas

export do_ML_spikes, make_data_and_fit, do_ML_filt_bound, do_ML_spikes_ΔLR, cross_validate_ΔLRmodel, compute_p0
export do_H_spikes, do_H_filt_bound, opt_func

function compute_p0(data,f_str,N;dt=1e-3);
    
    #### compute linear regression slope of tuning to $\Delta_{LR}$ and miniumum firing based on binning and averaging

    ΔLR = map((x,y,z)->diffLR(x,y,z,path=true,dt=dt),data["nT"],data["leftbups"],data["rightbups"]);
    nconds = 7;

    conds_bins = map(x->qcut(vcat(ΔLR[x]...),nconds,labels=false,duplicates="drop",retbins=true),data["trial"]);
    fr = map(j -> map(i -> (1/dt)*mean(vcat(data["spike_counts"][j]...)[conds_bins[j][1] .== i]),0:nconds-1),1:N);

    c0 = map((trial,k)->linreg(vcat(ΔLR[trial]...), vcat(k...)),data["trial"],data["spike_counts"]);

    if f_str == "exp"
        py0 = map((x,c0)->vcat(minimum(x),c0[2]),fr,c0);
    elseif f_str == "sig"
        py0 = map((x,c0)->vcat(minimum(x),maximum(x)-minimum(x),c0[2],0.),fr,c0);
    elseif f_str == "softplus"
        py0 = map((x,c0)->vcat(minimum(x),c0[2],0.),fr,c0);
    end
    
    return py0, fr
    
end

function make_data_and_fit(data,num_reps,fit_vec_z,
        dt,pzstar,pz0,model_type,map_str,f_str::String;all_sim::Bool=true,
        N::Int=0,noise::String="Poisson",
        betas::Vector{Vector{Float64}}=Vector{Vector{Float64}}(0),
        pystar::Union{Vector{Vector{Float64}},Vector{Float64}} = Vector{Vector{Float64}}(0),
        bias::Float64=0.,fity::Bool=false,fit_lambda::Bool=false,n::Int=103,
        x_tol::Float64=1e-16,f_tol::Float64=1e-16,g_tol::Float64=1e-12)   
    
    if model_type == "spikes"
        
        #make N neurons
        if all_sim
            data["N"] = map(x->x=collect(1:N),data["N"])
            data["N0"] = N;
        end

        #make fake data
        data = sampled_dataset!(data, pzstar, f_str; num_reps=num_reps, noise=noise, py=pystar, dt=dt);
        
        if fity
                
            fit_vec = cat(1,fit_vec_z,trues(length(pystar[1])*N));
            
            #organize by neuron to initalize p0
            data["trial"] = map(i -> (y = find(map(x -> any(x .== i), data["N"]) .== true); y[1]:y[end]),1:data["N0"])
            data["spike_counts"] = map(i -> map(j -> data["spike_counts"][j][i],data["trial"][i]),1:data["N0"]);
        
            py0, = compute_p0(data,f_str,N;dt=dt)
        
            mu0 = py0;
            
            if fit_lambda                      
                #convert back to trials
                data["spike_counts"] = map(i -> map(j -> data["spike_counts"][j][i],data["N"][i]),1:data["trial0"]);
                pz0, py1 = do_ML_filt_bound(pz0,deepcopy(py0),fit_vec,model_type,map_str,dt,data,betas,mu0,f_str)
            else
                py1 = do_ML_spikes_ΔLR(deepcopy(py0),data,map_str,betas,mu0,f_str,dt=dt);             
                #convert back to trials
                data["spike_counts"] = map(i -> map(j -> data["spike_counts"][j][i],data["N"][i]),1:data["trial0"]);
            end
                        
        else
            
            py1 = pystar;
            fit_vec = cat(1,fit_vec_z,falses(length(pystar[1])*N));
            
        end
        
        mu0 = py1;
        
        #fit and compute hessian
        pz, py = do_ML_spikes(copy(pz0),deepcopy(py1),fit_vec,model_type,map_str,dt,data,betas,mu0,noise,f_str,n=n,
            x_tol=x_tol,f_tol=f_tol,g_tol=g_tol)
        CI_z_plus, CI_z_minus, H = do_H_spikes(copy(pz),deepcopy(py),fit_vec,model_type,map_str,dt,data,betas,mu0,noise,f_str,n=n)
        
        #pz, py = do_ML_filt_bound(copy(pz0),deepcopy(py1),fit_vec,model_type,map_str,dt,data,betas,mu0,f_str)
        #CI_z_plus, CI_z_minus = do_H_filt_bound(copy(pz),deepcopy(py),fit_vec,model_type,map_str,dt,data,betas,mu0,f_str)
        
        return pz, CI_z_plus, CI_z_minus, py, data
                
    elseif model_type == "choice"
        
        #make fake data
        kwargs = ((:bias,bias))
        sampled_dataset!(data, pz, rng = 2, num_reps=num_reps; kwargs);
        
        p_z, CI_z_plus, CI_z_minus = do_ML_choice(pz_0,fit_vec,model_type,map_str,dt,data,bias)
        
    end

end

function do_ML_choice(p_z,fit_vec,model_type,map_str,dt,data,bias)
    
    ###########################################################################################
    ## Map parameters to unbounded domain for optimization

    p_z = inv_map_latent_params!(p_z,map_str,dt)  

    ###########################################################################################
    ## Concatenate into a single vector and break up into optimization variables and constants

    p = vcat(p_z,bias)

    p_opt = p[fit_vec]
    p_const = p[.!fit_vec];

    ###########################################################################################
    ## Optimize

    ll(x) = ll_wrapper(x, p_const, fit_vec, data, model_type, n=103);
    
    od = OnceDifferentiable(ll, p_opt; autodiff=:forward);
    
    p_opt = Optim.minimizer(Optim.optimize(od, p_opt, 
                BFGS(alphaguess = LineSearches.InitialStatic(alpha=1.0,scaled=true), 
                linesearch = BackTracking()), 
                Optim.Options(g_tol = 1e-12, x_tol = 1e-16, f_tol = 1e-16, 
                iterations = 1000, store_trace = true, 
                show_trace = false, extended_trace = false, allow_f_increases = true)));

    ###########################################################################################
    ## Break up optimization vector into functional groups and remap to bounded domain

    p_z,bias = latent_and_spike_params(p_opt, p_const, fit_vec, model_type)
    p_z = map_latent_params!(p_z,map_str,dt)   
    
    ###########################################################################################
    ## compute Hessian

    H = ForwardDiff.hessian(ll, p_opt);
    d,V = eig(H)
    
    if all(d .> 0)
    
        CI = 2*sqrt.(diag(inv(H)));
    
        CI_z_plus, CI_bias_plus = latent_and_spike_params(p_opt + CI, p_const, fit_vec, model_type)
        CI_z_plus = map_latent_params!(CI_z_plus,map_str,dt) 

        CI_z_minus, CI_bias_minus = latent_and_spike_params(p_opt - CI, p_const, fit_vec, model_type)
        CI_z_minus = map_latent_params!(CI_z_minus,map_str,dt)
    else
        
        CI_z_plus = similar(p_z);
        CI_z_minus = similar(p_z);
        
    end
    
    return p_z, CI_z_plus, CI_z_minus
    
end

function do_H_spikes(pz,py,fit_vec,model_type,map_str,dt,data,beta,mu0,noise,f_str;n::Int=103)
    
    ###########################################################################################
    ## Map parameters to unbounded domain for optimization

    pz = inv_map_latent_params!(pz,map_str,dt)    
    py = map(x->inv_map_lambda_y_p!(x,f_str;map_str=map_str),py)

    ###########################################################################################
    ## Concatenate into a single vector and break up into optimization variables and constants

    p = vcat(pz,vcat(py...))
    p_opt = p[fit_vec]
    p_const = p[.!fit_vec];

    ###########################################################################################
    ## Optimize

    ll(x) = ll_wrapper(x, p_const, fit_vec, data, model_type, f_str, 
        beta=beta, mu0=mu0, noise=noise, n=n, dt=dt);

    ###########################################################################################
    ## Break up optimization vector into functional groups and remap to bounded domain

    pz,py = latent_and_spike_params(p_opt, p_const, fit_vec, model_type, f_str)
    pz = map_latent_params!(pz,map_str,dt)    
    py = map(x->map_lambda_y_p!(x,f_str;map_str=map_str),py)
    
    ###########################################################################################
    ## compute Hessian

    H = ForwardDiff.hessian(ll, p_opt);
    d,V = eig(H)
    
    if all(d .> 0)
    
        CI = 2*sqrt.(diag(inv(H)));
    
        CI_z_plus, CI_y_plus = latent_and_spike_params(p_opt + CI, p_const, fit_vec, model_type, f_str)
        CI_z_plus = map_latent_params!(CI_z_plus,map_str,dt) 
        #CI_y_plus = map((x)->map_sig_params!(x,map_str), CI_y_plus)

        CI_z_minus, CI_y_minus = latent_and_spike_params(p_opt - CI, p_const, fit_vec, model_type, f_str)
        CI_z_minus = map_latent_params!(CI_z_minus,map_str,dt)
        #CI_y_minus = map((x)->map_sig_params!(x,map_str),CI_y_minus);
    else
        
        CI_z_plus = similar(pz);
        CI_z_minus = similar(pz);
        
    end
    
    return CI_z_plus, CI_z_minus, H
    
end

function do_ML_spikes(pz,py,fit_vec,model_type,map_str,dt,data,beta,mu0,noise,f_str;n::Int=103,
    x_tol::Float64=1e-16,f_tol::Float64=1e-16,g_tol::Float64=1e-12)
    
    ###########################################################################################
    ## Map parameters to unbounded domain for optimization

    pz = inv_map_latent_params!(pz,map_str,dt)     
    py = map(x->inv_map_lambda_y_p!(x,f_str;map_str=map_str),py)

    ###########################################################################################
    ## Concatenate into a single vector and break up into optimization variables and constants

    p = vcat(pz,vcat(py...))
    p_opt = p[fit_vec]
    p_const = p[.!fit_vec];

    ###########################################################################################
    ## Optimize

    ll(x) = ll_wrapper(x, p_const, fit_vec, data, model_type, f_str, 
        beta=beta, mu0=mu0, noise=noise, n=n, dt=dt);
    
    od = OnceDifferentiable(ll, p_opt; autodiff=:forward);
    
    p_opt = Optim.minimizer(Optim.optimize(od, p_opt, 
                BFGS(alphaguess = LineSearches.InitialStatic(alpha=1.0,scaled=true), 
                linesearch = BackTracking()), 
                Optim.Options(g_tol = g_tol, x_tol = x_tol, f_tol = f_tol, 
                iterations = 1000, store_trace = true, 
                show_trace = true, extended_trace = false, allow_f_increases = true)));

    ###########################################################################################
    ## Break up optimization vector into functional groups and remap to bounded domain

    pz,py = latent_and_spike_params(p_opt, p_const, fit_vec, model_type, f_str)
    pz = map_latent_params!(pz,map_str,dt)       
    py = map(x->map_lambda_y_p!(x,f_str;map_str=map_str),py)
    
    return pz, py
    
end

function do_H_filt_bound(pz,py,fit_vec,model_type,map_str,dt,data,beta,mu0,f_str,noise)
    
    ###########################################################################################
    ## Map parameters to unbounded domain for optimization

    pz = inv_map_latent_params!(pz,map_str,dt)    
    py = map(x->inv_map_lambda_y_p!(x,f_str;map_str=map_str),py)

    ###########################################################################################
    ## Concatenate into a single vector and break up into optimization variables and constants

    p = vcat(pz,vcat(py...))
    p_opt = p[fit_vec]
    p_const = p[.!fit_vec];

    ###########################################################################################
    ## Closure over function

    ll(x) = filt_bound_model_wrapper(x, p_const, fit_vec, data, model_type, f_str,
        beta=beta, mu0=mu0, dt=dt, noise=noise);

    ###########################################################################################
    ## Break up optimization vector into functional groups and remap to bounded domain

    pz,py = latent_and_spike_params(p_opt, p_const, fit_vec, model_type, f_str)
    pz = map_latent_params!(pz,map_str,dt)    
    py = map(x->map_lambda_y_p!(x,f_str;map_str=map_str),py)
    
    ###########################################################################################
    ## compute Hessian

    H = ForwardDiff.hessian(ll, p_opt);
    d,V = eig(H)
    
    if all(d .> 0)
    
        CI = 2*sqrt.(diag(inv(H)));
    
        CI_z_plus, CI_y_plus = latent_and_spike_params(p_opt + CI, p_const, fit_vec, model_type, f_str)
        CI_z_plus = map_latent_params!(CI_z_plus,map_str,dt) 
        #CI_y_plus = map((x)->map_sig_params!(x,map_str), CI_y_plus)

        CI_z_minus, CI_y_minus = latent_and_spike_params(p_opt - CI, p_const, fit_vec, model_type, f_str)
        CI_z_minus = map_latent_params!(CI_z_minus,map_str,dt)
        #CI_y_minus = map((x)->map_sig_params!(x,map_str),CI_y_minus);
    else
        
        CI_z_plus = similar(pz);
        CI_z_minus = similar(pz);
        
    end
    
    return CI_z_plus, CI_z_minus, H
    
end

function do_ML_filt_bound(pz,py,fit_vec,model_type,map_str,dt,data,beta,mu0,f_str,noise)
    
    ###########################################################################################
    ## Map parameters to unbounded domain for optimization

    pz = inv_map_latent_params!(pz,map_str,dt)    
    py = map(x->inv_map_lambda_y_p!(x,f_str;map_str=map_str),py)

    ###########################################################################################
    ## Concatenate into a single vector and break up into optimization variables and constants

    p = vcat(pz,vcat(py...))
    p_opt = p[fit_vec]
    p_const = p[.!fit_vec];

    ###########################################################################################
    ## Optimize

    ll(x) = filt_bound_model_wrapper(x, p_const, fit_vec, data, model_type, f_str,
        beta=beta, mu0=mu0, dt=dt, noise=noise);
    
    od = OnceDifferentiable(ll, p_opt; autodiff=:forward);
    
    p_opt = Optim.minimizer(Optim.optimize(od, p_opt, 
                BFGS(alphaguess = LineSearches.InitialStatic(alpha=1.0,scaled=true), 
                linesearch = BackTracking()), 
                Optim.Options(g_tol = 1e-12, x_tol = 1e-16, f_tol = 1e-16, 
                iterations = 1000, store_trace = true, 
                show_trace = true, extended_trace = false, allow_f_increases = true)));

    ###########################################################################################
    ## Break up optimization vector into functional groups and remap to bounded domain

    pz,py = latent_and_spike_params(p_opt, p_const, fit_vec, model_type, f_str)
    pz = map_latent_params!(pz,map_str,dt)   
    py = map(x->map_lambda_y_p!(x,f_str;map_str=map_str),py) 
    
    return pz,py
    
end

function do_ML_spikes_ΔLR(py::Vector{Vector{Float64}},data,map_str::String,
        betas::Vector{Vector{Float64}},mu0::Vector{Vector{Float64}},f_str::String;
        dt::Float64=1e-3,std0::Vector{Float64}=Vector{Float64}(),noise::String="Poisson")
    
    ΔLR = map((x,y,z)->diffLR(x,y,z,path=true,dt=dt),data["nT"],data["leftbups"],data["rightbups"]);
    
    py = map(x->inv_map_lambda_y_p!(x,f_str;map_str=map_str),py)
    
    if noise == "Gaussian"
    #optimize all neurons
        pystar = pmap((py,k,trials,mu0,beta,std0)->opt_func(py,k,trials,ΔLR,f_str,map_str=map_str,dt=dt,
            beta=beta,mu0=mu0,std0=std0,noise=noise),py,data["spike_counts"],data["trial"],mu0,betas,std0);
    elseif noise == "Poisson"
        pystar = pmap((py,k,trials,mu0,beta,std0)->opt_func(py,k,trials,ΔLR,f_str,map_str=map_str,dt=dt,
            beta=beta,mu0=mu0,noise=noise),py,data["spike_counts"],data["trial"],mu0,betas);
    end
    
    #map pystar parameters  
    pystar = map(x->map_lambda_y_p!(x,f_str;map_str=map_str),pystar)
    
    return pystar
    
end

function opt_func(p0,k,trials,ΔLR,f_str::String;map_str::String="exp",dt::Float64=1e-3,
    beta::Vector{Float64}=Vector{Float64}(), mu0::Vector{Float64}=Vector{Float64}(), std0::Float64=1e-1,
    noise::String="Poisson")
    
    Optim.minimizer(optimize(p0 -> ΔLR_ll_wrapper(p0,k,ΔLR[trials],f_str,map_str=map_str,dt=dt,beta=beta,mu0=mu0,
            noise=noise,std0=std0), 
            p0, method = Optim.BFGS(alphaguess = InitialStatic(alpha=1.0,scaled=true), 
            linesearch = BackTracking()), autodiff=:forward, g_tol = 1e-12, x_tol = 1e-16, 
            f_tol = 1e-16, iterations = Int(1e16), show_trace = false, allow_f_increases = true));
    
end

function cross_validate_ΔLRmodel(py,k,trials,ΔLR,ntrials,Beta,f_str::String;map_str="exp",dt=1e-3,rng=1)
    
    #For fitting the perfect accumulator no bound latent and cross validating
    
    idx_train = sort(sample(srand(rng),1:length(k),ntrials,replace=false))
    
    if isempty(setdiff(1:length(k),idx_train))
        idx_test = idx_train
    else
        idx_test = setdiff(1:length(k),idx_train)
    end

    k_train = k[idx_train]
    trials_train = trials[idx_train]

    k_test = k[idx_test]
    trials_test = trials[idx_test]
    
    #map f.r. parameters to unbounded domain
    py = inv_map_lambda_y_p!(py,f_str;map_str=map_str)
    
    #set all betas to the input provided value
    beta = vcat(Beta*ones(length(py)))
        
    #set the mean of the prior to the initiali parameters
    mu0 = py;
            
    #max LL for training data
    pystar = opt_func(copy(py),k_train,trials_train,ΔLR,f_str,map_str=map_str,dt=dt,beta=beta,mu0=mu0)
    
    #be careful of this, note that when this wrapper function is not used in the context of an optimization
    # the input py is mutuable.
    #compute test LL
    LL_CV = ΔLR_ll_wrapper(copy(pystar),k_test,ΔLR[trials_test],f_str,map_str=map_str,dt=dt,beta=beta,mu0=mu0)
       
    return LL_CV
    
end

function my_callback(os)

    so_far = time() - start_time
    println(" * Time so far:     ", so_far)
  
    history = Array{Float64,2}(sum(fit_vec),0)
    history_gx = Array{Float64,2}(sum(fit_vec),0)
    for i = 1:length(os)
        ptemp = group_params(os[i].metadata["x"], p_const, fit_vec)
        ptemp = map_func!(ptemp,model_type,"tanh",N=N)
        ptemp_opt, = break_params(ptemp, fit_vec)       
        history = cat(2,history,ptemp_opt)
        history_gx = cat(2,history_gx,os[i].metadata["g(x)"])
    end
    save(out_pth*"/history.jld", "history", history, "history_gx", history_gx)

    return false

end

end