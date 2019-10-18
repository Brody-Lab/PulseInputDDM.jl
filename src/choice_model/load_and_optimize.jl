#################################### Choice observation model #################################

function generate_syn_data_fit_CI(pz::Dict{}, pd::Dict{}, ntrials::Int;
        n::Int=53, dt=1e-2, use_bin_center::Bool=false)

    data = sample_inputs_and_choices(pz["generative"], pd["generative"], ntrials;
        use_bin_center=use_bin_center)

    data = bin_clicks!(data,use_bin_center;dt=dt)

    pz, pd, = optimize_model(pz, pd, data; n=n, show_trace=true)
    pz, pd = compute_H_CI!(pz, pd, data; n=n)

end

function load_and_optimize(path::String, sessids, ratnames; n::Int=53, dt::Float64=1e-2,
        pz::Dict = Dict("name" => ["σ_i","B", "λ", "σ_a","σ_s","ϕ","τ_ϕ"],
        "fit" => trues(dimz),
        "initial" => [1e-6,20.,-0.1,100.,5.,0.2,0.005],
        "lb" => [eps(), 4., -5., eps(), eps(), eps(), eps()],
        "ub" => [10., 100, 5., 800., 40., 2., 10.]),
        show_trace::Bool=true, iterations::Int=Int(2e3),
        x_tol::Float64=1e-4, f_tol::Float64=1e-9, g_tol::Float64=1e-2)

    data = aggregate_choice_data(path,sessids,ratnames)
    data = bin_clicks!(data,use_bin_center;dt=dt)

    pz, pd = load_and_optimize(data; n=n,
        pz=pz, show_trace=show_trace, iterations=iterations)

    return pz, pd

end

function load_and_optimize(data; n::Int=53,
        pz::Dict = Dict("name" => ["σ_i","B", "λ", "σ_a","σ_s","ϕ","τ_ϕ"],
        "fit" => trues(dimz),
        "initial" => [1e-6,20.,-0.1,100.,5.,0.2,0.005],
        "lb" => [eps(), 4., -5., eps(), eps(), eps(), eps()],
        "ub" => [10., 100, 5., 800., 40., 2., 10.]),
        show_trace::Bool=true, iterations::Int=Int(2e3),
        x_tol::Float64=1e-4, f_tol::Float64=1e-9, g_tol::Float64=1e-2)

    #parameters for the choice observation
    pd = Dict("name" => vcat("bias","lapse"), "fit" => trues(2),
        "initial" => vcat(0.,0.5))

    pz, pd = optimize_model(pz, pd, data; n=n, show_trace=show_trace, iterations=iterations)

    return pz, pd

end
