module ML_and_CL

using latent_model, LineSearches, Optim, global_functions, ForwardDiff
using sample_model

export do_ML, make_data_and_fit

function make_data_and_fit(data,num_reps,fit_vec_z,
        dt,pz,model_type,map_str;all_sim::Bool=true,
        N::Int=0,noise::String="Poisson",dimy::Int=4,
        beta_y::Vector{Float64}=Vector{Float64}(0),
        pyp::Union{Vector{Vector{Float64}},Vector{Float64}} = Vector{Vector{Float64}}(0),
        bias::Float64=0.)   
        
    pz_0 = copy(pz);
    #pz =            [10.,      0.,         10., -2.,        40.,    1e-6,      1.,    0.2];
    #p_z_0_prime =    [1.,      0.,         5., -5.,           1.,    0.1,      1.,    0.2];
    #pz_0[fit_vec_z] = p_z_0_prime[fit_vec_z];
    
    if model_type == "spikes"
        #make N neurons
        all_sim ? data["N"] = map(x->x=collect(1:N),data["N"]) : nothing
        
        fit_vec = cat(1,fit_vec_z,falses(dimy*N));
        #fit_vec = cat(1,fit_vec_z,trues(dimy*N));

        #make the firing rate parameters
        eltype(pyp) == Float64 ? py = map(i->copy(pyp),1:N) : py = deepcopy(pyp);

        #define the means for the priors
        #mu0_y = map(x->vcat(x,zeros(3)),zeros(N));
        mu0_y = py;

        #make fake data
        kwargs = ((:py,py))
        sampled_dataset!(data, pz, rng = 2, num_reps=num_reps, noise=noise; kwargs);
        
        #fit and compute hessian
        p_z, CI_z_plus, CI_z_minus = do_ML(pz_0,py,fit_vec,model_type,map_str,dt,data,beta_y,mu0_y,noise)
        
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

function do_ML(p_z,p_y,fit_vec,model_type,map_str,dt,data,beta_y,mu0_y,noise)
    
    ###########################################################################################
    ## Map parameters to unbounded domain for optimization

    p_z = inv_map_latent_params!(p_z,map_str,dt)  
    p_y = map(x->inv_map_sig_params!(x,map_str),p_y);

    ###########################################################################################
    ## Concatenate into a single vector and break up into optimization variables and constants

    p = vcat(p_z,vcat(p_y...))

    p_opt = p[fit_vec]
    p_const = p[.!fit_vec];

    ###########################################################################################
    ## Optimize

    ll(x) = ll_wrapper(x, p_const, fit_vec, data, model_type,
        beta_y=beta_y, mu0_y=mu0_y, noise=noise, n=103);
    
    od = OnceDifferentiable(ll, p_opt; autodiff=:forward);
    
    p_opt = Optim.minimizer(Optim.optimize(od, p_opt, 
                BFGS(alphaguess = LineSearches.InitialStatic(alpha=1.0,scaled=true), 
                linesearch = BackTracking()), 
                Optim.Options(g_tol = 1e-12, x_tol = 1e-16, f_tol = 1e-16, 
                iterations = 1000, store_trace = true, 
                show_trace = true, extended_trace = false, allow_f_increases = true)));

    ###########################################################################################
    ## Break up optimization vector into functional groups and remap to bounded domain

    p_z,p_y = latent_and_spike_params(p_opt, p_const, fit_vec, model_type)
    p_z = map_latent_params!(p_z,map_str,dt)   
    p_y = map(x->map_sig_params!(x,map_str),p_y);
    
    ###########################################################################################
    ## compute Hessian

    H = ForwardDiff.hessian(ll, p_opt);
    d,V = eig(H)
    
    if all(d .> 0)
    
        CI = 2*sqrt.(diag(inv(H)));
    
        CI_z_plus, CI_y_plus = latent_and_spike_params(p_opt + CI, p_const, fit_vec, model_type)
        CI_z_plus = map_latent_params!(CI_z_plus,map_str,dt) 
        CI_y_plus = map((x)->map_sig_params!(x,map_str), CI_y_plus)

        CI_z_minus, CI_y_minus = latent_and_spike_params(p_opt - CI, p_const, fit_vec, model_type)
        CI_z_minus = map_latent_params!(CI_z_minus,map_str,dt)
        CI_y_minus = map((x)->map_sig_params!(x,map_str),CI_y_minus);
    else
        
        CI_z_plus = similar(p_z);
        CI_z_minus = similar(p_z);
        
    end
    
    return p_z, CI_z_plus, CI_z_minus
    
end

end