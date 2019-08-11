#################################### Full Poisson neural observation model #############

function generate_syn_data_fit_CI(n::Int, pz::Dict{}, py::Dict{}, ntrials_per_sess; 
        dt=1e-3, f_str::String="softplus",use_bin_center::Bool=true)
   
    data = sample_input_and_spikes_multiple_sessions(n, pz["generative"], py["generative"], ntrials_per_sess; f_str=f_str,
        dtMC=1e-4,use_bin_center=false)
    
    @warn "data generated using FP @ dtMC=1e-4, which will introduce lots of settling!"
    
    data = map(x->bin_clicks_spikes_and_λ0!(x,use_bin_center;dt=dt), data)
    
    pz["initial"][pz["fit"] .== false] = pz["generative"][pz["fit"] .== false]
    #py["initial"][py["fit"] .== false] = py["generative"][py["fit"] .== false]
    
    pz, py, = optimize_model(pz, py, data, f_str, n, show_trace=true)   
    pz, py = compute_H_CI!(pz, py, data, f_str, n)
    
end


function generate_syn_data_fit_CI(pz::Dict{}, py::Dict{}, ntrials_per_sess; 
        n::Int=203, dt=1e-3, f_str::String="softplus", rng::Int=0,use_bin_center::Bool=true)
   
    data = sample_input_and_spikes_multiple_sessions(pz["generative"], py["generative"], ntrials_per_sess; f_str=f_str,
        rng=rng,use_bin_center=false)
    
    data = map(x->bin_clicks_spikes_and_λ0!(x,use_bin_center;dt=dt), data)
    
    pz["initial"][pz["fit"] .== false] = pz["generative"][pz["fit"] .== false]
    #py["initial"][py["fit"] .== false] = py["generative"][py["fit"] .== false]
    
    if n == 0
        pz, py, = optimize_model(pz, py, data, f_str, show_trace=true)  
        pz, py = compute_H_CI!(pz, py, data, f_str)
    else
        pz, py, = optimize_model(pz, py, data, f_str, n, show_trace=true)   
        pz, py = compute_H_CI!(pz, py, data, f_str, n)
    end
    
end

function generate_syn_data_fit_CI(nsessions, N_per_sess, ntrials_per_sess; 
        n::Int=53, dt=1e-2, dimy::Int=3, f_str::String="softplus",
        pz::Dict = Dict("generative" => [0., 15., -1., 100., 0., 1., 0.02], 
        "name" => ["σ_i","B", "λ", "σ_a","σ_s","ϕ","τ_ϕ"],
        "fit" => vcat(falses(1),trues(3),falses(3)),
        "initial" => [2.,20.,-3,100.,2.,0.2,0.005],
        "lb" => [-eps(), 4., -5., -eps(), -eps(), eps(), eps()],
        "ub" => [10., 100, 5., 800., 40., 2., 10.]),
        use_bin_center::Bool=true)    
    
    pz["initial"][pz["fit"] .== false] = pz["generative"][pz["fit"] .== false]
    #pz["initial"] = pz["generative"]

    #py = Dict("generative" => [[[1e-6, 10., 1e-6] for n in 1:N] for N in N_per_sess], 
    #    "fit" => map(N-> repeat([falses(3)],outer=N), N_per_sess),
    #    "initial" => [[[Vector{Float64}(undef,dimy)] for n in 1:N] for N in N_per_sess],
    #    "dimy"=> dimy,
    #    "N"=> N_per_sess,
    #    "nsessions"=> nsessions)
        
    py = Dict("generative" => [[[10., 10., -10.] for n in 1:N] for N in N_per_sess], 
        "fit" => map(N-> repeat([falses(1),trues(1),falses(1)],outer=N), N_per_sess),
        "initial" => [[[Vector{Float64}(undef,dimy)] for n in 1:N] for N in N_per_sess],
        "dimy"=> dimy,
        "N"=> N_per_sess,
        "nsessions"=> nsessions)
   
    data = sample_input_and_spikes_multiple_sessions(pz["generative"], py["generative"], ntrials_per_sess;
        use_bin_center=false)
    
    data = map(x->bin_clicks_spikes_and_λ0!(x,use_bin_center;dt=dt), data)
    
    #py["initial"] = map(data-> regress_init(data, f_str), data)
    #py["initial"] = map(data -> optimize_model(data, f_str, show_trace=false), data)
    py["initial"] = py["generative"]
    
    if n == 0
        pz, py, = optimize_model(pz, py, data, f_str, show_trace=false)  
        pz, py = compute_H_CI!(pz, py, data, f_str)
    else
        pz, py, = optimize_model(pz, py, data, f_str, n, show_trace=true)   
        pz, py = compute_H_CI!(pz, py, data, f_str, n)
    end
    
end

function load_and_optimize(path::String, sessids, ratnames, f_str, n::Int; 
        dt::Float64=1e-3, delay::Float64=0.,
        pz::Dict = Dict("name" => ["σ_i","B", "λ", "σ_a","σ_s","ϕ","τ_ϕ"],
            "fit" => vcat(falses(1),trues(2),falses(4)),
            "initial" => [2*eps(), 10., -0.1, 2*eps(), 2*eps(), 1.0, 0.005],
            "lb" => [eps(), 8., -5., eps(), eps(), eps(), eps()],
            "ub" => [10., 40, 5., 100., 2.5, 2., 10.]),
        show_trace::Bool=true, iterations::Int=Int(2e3),
        use_bin_center::Bool=true)
    
    data = aggregate_spiking_data(path,sessids,ratnames)
    data = map(x->bin_clicks_spikes_and_λ0!(x,use_bin_center; dt=dt,delay=delay), data)
    
    pz, py = load_and_optimize(data, f_str, n; pz=pz, 
        show_trace=show_trace, iterations=iterations)
    
    return pz, py, data
    
end

function load_and_optimize(data::Vector{Dict{Any,Any}}, f_str, n::Int;
        pz::Dict = Dict("name" => ["σ_i","B", "λ", "σ_a","σ_s","ϕ","τ_ϕ"],
            "fit" => vcat(falses(1),trues(2),falses(4)),
            "initial" => [2*eps(), 10., -0.1, 2*eps(), 2*eps(), 1.0, 0.005],
            "lb" => [eps(), 8., -5., eps(), eps(), eps(), eps()],
            "ub" => [10., 40, 5., 100., 2.5, 2., 10.]),
        show_trace::Bool=true, iterations::Int=Int(2e3))
    
    nsessions = length(data)
    N_per_sess = map(data-> data["N"], data)
    
    if f_str == "softplus"
        dimy = 3
    elseif f_str == "sig"
        dimy = 4
    end

    #I should map over this, no map within this....
    #parameters for the neural observation model
    py = Dict("fit" => map(N-> repeat([trues(dimy)],outer=N), N_per_sess),
        "initial" => [[[Vector{Float64}(undef,dimy)] for n in 1:N] for N in N_per_sess],
        "dimy"=> dimy,
        "N"=> N_per_sess,
        "nsessions"=> nsessions)

    py["initial"] = map(data-> regress_init(data, f_str), data)
    pz, py = optimize_model(pz, py, data, f_str, show_trace=show_trace, iterations=iterations)
    
    pz["initial"] = vcat(1.,10.,-0.1,20.,0.5,1.0,0.005)
    pz["state"][pz["fit"] .== false] = pz["initial"][pz["fit"] .== false]
    pz["fit"] = vcat(trues(5),falses(2))
    
    pz, py, = optimize_model(pz, py, data, f_str, n, show_trace=show_trace, iterations=iterations) 
    pz, py = compute_H_CI!(pz, py, data, f_str, n)
    
    LL_ML = compute_LL(pz["final"], py["final"], data, n, f_str)

    LL_null = mapreduce(d-> mapreduce(r-> mapreduce(n-> 
                neural_null(d["spike_counts"][r][n], d["λ0"][r][n], d["dt"]), 
                    +, 1:d["N"]), +, 1:d["ntrials"]), +, data)

    ΔLL = LL_ML - LL_null
   
    return pz, py, ΔLL
    
end