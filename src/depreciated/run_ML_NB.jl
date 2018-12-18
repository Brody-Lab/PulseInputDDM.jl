
addprocs(24);

using latent_model, LineSearches, Optim, JLD, ForwardDiff
using global_functions, initialize_latent_model

ratname = "T036"
model_type_dir = "spikes"
model_type_dir == "joint" ? model_type = ["spikes","choice"] : model_type = model_type_dir
path=ENV["HOME"]*"/Dropbox/spike-data_latent-accum";
reload_pth = path*"/data/results/julia/"*model_type_dir*"/"*ratname
out_pth = reload_pth;
max_time = 2; # time limit in hours

model_type_dir == "joint" ? model_type = ["spikes","choice"] : model_type = model_type_dir
map_str = "exp"
dt = 2e-2;

## Load data

if any(model_type .== "spikes") & any(model_type .== "choice")

    fit_vec, data, model_type, p0_z, p0_bias, p0_y, beta_y, mu0_y = 
        load_data(path,model_type,reload_pth,map_str,ratname);

elseif any(model_type .== "spikes") 

    fit_vec, data, model_type, p0_z, p0_y, beta_y, mu0_y = 
        load_data(path,model_type,reload_pth,map_str,ratname);

elseif any(model_type .== "choice")

    fit_vec, data, model_type, p0_z, p0_bias = 
        load_data(path,model_type,reload_pth,map_str,ratname);

end

## Load previous results

results = reload_pth*"/results.jld"

if isfile(results)
    
    p_z = load(results, "p_z")

    if any(model_type .== "spikes") 

        p_y = load(results, "p_y")

    elseif any(model_type .== "choice")

        p_bias = load(results, "p_bias")

    end

else
    
    p_z = p0_z

    if any(model_type .== "spikes") 

        p_y = p0_y

    elseif any(model_type .== "choice")

        p_bias = p0_bias

    end

end

#this is old stuff
#tail="18359421";
#reload_pth2=path*"/data/results/julia/"*model_type*"/"*ratname*"/"*tail;
#p_opt,x0,fit_vec,betas,mu0 = load(reload_pth2*"/results.jld","p_opt","x0","fit_vec","betas","mu0");
#p_opt[7:end] = reshape(reshape(p_opt[7:end],12,4)',48,1);
#p_z,p_y = latent_and_spike_params(p_opt, x0[.!fit_vec], fit_vec, model_type)

## Map parameters to unbounded domain for optimization

p_z = inv_map_latent_params!(p_z,map_str,dt)  
any(model_type .== "spikes") ? p_y = map(x->inv_map_sig_params!(x,map_str),p_y) : nothing

## Concatenate into a single vector and break up into optimization variables and constants

if any(model_type .== "spikes") & any(model_type .== "choice")

    p = vcat(p_z,p_bias,vcat(p_y...))

elseif any(model_type .== "spikes") 

    p = vcat(p_z,vcat(p_y...))

elseif any(model_type .== "choice")

    p = vcat(p_z,p_bias)

end

p_opt = p[fit_vec]
p_const = p[.!fit_vec]

## Define a closure over optimization function

if any(model_type .== "spikes") 

    @everywhere ll(x) = ll_wrapper(x, p_const, fit_vec, data, model_type,
        beta_y=beta_y, mu0_y=mu0_y);
    
else
    
    @everywhere ll(x) = ll_wrapper(x, p_const, fit_vec, data, model_type);
    
end

od = OnceDifferentiable(ll, p_opt; autodiff=:forward);

## Optimize and compute Hessian

@time p_opt = Optim.minimizer(Optim.optimize(od, p_opt, 
            BFGS(alphaguess = LineSearches.InitialStatic(alpha=1.0,scaled=true), 
            linesearch = BackTracking()), 
            Optim.Options(time_limit = 3600 * 0.9 * max_time, 
            g_tol = 1e-12, x_tol = 1e-16, f_tol = 1e-16, 
            iterations = 10, store_trace = true, 
            show_trace = true, extended_trace = false, allow_f_increases = true)))

H = ForwardDiff.hessian(od, p_opt)

## Break up optimization vector into functional groups and remap to bounded domain

if any(model_type .== "spikes") & any(model_type .== "choice")

    p_z,p_y,p_bias = latent_and_spike_params(p_opt, p_const, fit_vec, model_type)
    p_z = map_latent_params!(p_z,map_str,dt)   
    p_y = map(x->map_sig_params!(x,map_str),p_y)

elseif any(model_type .== "spikes") 

    p_z,p_y = latent_and_spike_params(p_opt, p_const, fit_vec, model_type)
    p_z = map_latent_params!(p_z,map_str,dt)   
    p_y = map(x->map_sig_params!(x,map_str),p_y)

elseif any(model_type .== "choice")

    p_z,p_bias = latent_and_spike_params(p_opt, p_const, fit_vec, model_type)
    p_z = map_latent_params!(p_z,map_str,dt)   

end

## save ML parameters

if any(model_type .== "spikes") & any(model_type .== "choice")

    save(out_pth*"/results.jld","p_z", p_z, "p_bias", p_bias, "p_y", p_y) 

elseif any(model_type .== "spikes") 

    save(out_pth*"/results.jld","p_z", p_z, "p_y", p_y) 

elseif any(model_type .== "choice")

    save(out_pth*"/results.jld","p_z", p_z, "p_bias", p_bias) 

end

## save Hessian

save(out_pth*"/Hessian.jld","H", H) 
