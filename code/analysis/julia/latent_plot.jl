function latent_plot(reload_pth)
    
    using PyPlot

    p_opt,x0,N,fit_vec,data,sessid,betas,mu0 = 
        load(reload_pth*"/results.jld","p_opt","x0","N","fit_vec","data","sessid","betas","mu0");

    Hess = load(reload_pth*"/Hessian.jld","H");

    model_type == "joint" ? (model_type = ["choice", "spikes"];) : nothing

    p_const = x0[.!fit_vec];
    p_CI = 2*sqrt.(diag(inv(Hess)))
    p = group_params(p_opt, p_const, fit_vec)

    num_rows, num_cols = 2, 3
    fig, axes = subplots(num_rows, num_cols, figsize=(16,6))
    subplot_num = 0

    param_str = ["B","λ","log(σ_a)","log(σ_s)","ϕ","τ_ϕ"]
    plot_vec = find(fit_vec(1:8));

    for i in 1:num_rows
        for j in 1:num_cols
            ax = axes[i, j]
            subplot_num += 1
            ax[:errorbar](1, p[plot_vec[subplot_num]], yerr = p_CI[subplot_num],fmt="o")
            ax[:set_title]("$(param_str[subplot_num])",fontsize=10)
        end
    end

end