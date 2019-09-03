using Revise, Distributed

addprocs(44);
@everywhere using Pkg; @everywhere Pkg.activate("/usr/people/briandd/Projects/pulse_input_DDM.jl")

@everywhere using pulse_input_DDM
using JLD2, Statistics

path = ENV["HOME"]*"/Projects/pulse_input_DDM.jl"

region = ARGS[1]
sessids, ratnames = sessids_from_region(region);

idx = parse(Int,ARGS[2])
rat = ratnames[idx]
sesss = sessids[idx]

model = ARGS[3]

n, dt = 103, 1e-2

if model == "neural"
    
    data = aggregate_spiking_data(path*"/data/hanks_data_sessions/all_times", [sesss], [rat]);

    if (region == "STR") || (region == "FOF")
        delay = 0.05
    elseif region == "PPC"
        delay = 0.1
    end
    
    use_bin_center = true;
    data = map(x->bin_clicks_spikes_and_λ0!(x, use_bin_center; dt=dt,delay=delay), data);

    nsessions, N_per_sess, dimy, f_str = length(data), map(data-> data["N"], data), 4, "sig"
    
    if isfile(path*"/data/results/working/by_rat/softplus_sig/w_adapt/"*region*"_"*rat*".jld")
        JLD2.@load path*"/data/results/working/by_rat/softplus_sig/w_adapt/"*region*"_"*rat*".jld" pz py
        
    else
        
        pz = Dict("name" => ["σ_i","B", "λ", "σ_a","σ_s","ϕ","τ_ϕ"],
            "fit" => vcat(falses(1),trues(2),falses(4)),
            "initial" => [1., 10., -0.1, 2*eps(), 2*eps(), 1.0-eps(), 0.005],
            "lb" => [eps(), 8., -5., eps(), eps(), eps(), eps()],
            "ub" => [10., 40, 5., 100., 2.5, 2., 1.])

        py = Dict("fit" => map(N-> repeat([trues(dimy)],outer=N), N_per_sess),
            "initial" => [[[Vector{Float64}(undef,dimy)] for n in 1:N] for N in N_per_sess],
            "dimy"=> dimy,
            "N"=> N_per_sess,
            "nsessions"=> nsessions)

        py["initial"] = map(data-> regress_init(data, f_str), data)
        #pz["fit"] =  vcat(falses(1),trues(2),falses(4))
        pz, py = optimize_model(pz, py, data, f_str, show_trace=true, iterations=200)

        pz["initial"] = vcat(1.,10.,-0.1,20.,0.5,1.0-eps(),0.005)
        pz["state"][pz["fit"] .== false] = pz["initial"][pz["fit"] .== false]
        pz["fit"] = vcat(trues(7))
        
    end

    @time pz, py, opt_output = optimize_model(pz, py, data, f_str, n, show_trace=true, iterations=500) 
    #check opt_output, to determine if should compute the Hessian
    @time pz, py, H = compute_H_CI!(pz, py, data, f_str, n)

    LL_ML = compute_LL(pz["final"], py["final"], data, n, f_str)

    LL_null = mapreduce(d-> mapreduce(r-> mapreduce(n-> 
                neural_null(d["spike_counts"][r][n], d["λ0"][r][n], d["dt"]), 
                    +, 1:d["N"]), +, 1:d["ntrials"]), +, data)

    ΔLL = LL_ML - LL_null

    #add try catch
    μ_hat_ct = pulse_input_DDM.sample_average_expected_rates_multiple_sessions(pz["final"], 
        py["final"], data, f_str)

    JLD2.@save path*"/data/results/working/by_rat/softplus_sig/w_adapt/"*region*"_"*rat*".jld" pz py ΔLL μ_hat_ct H data
    #BSON.@save path*"/data/results/working/by_rat/softplus_sig/w_adapt/"*region*"_"*rat*".bson" pz py ΔLL μ_hat_ct data
    
    
elseif model == "neural_break"
    
    #data = aggregate_spiking_data(path*"/data/hanks_data_sessions/all_times", [sesss], [rat]);
    data = aggregate_spiking_data(path*"/data/hanks_data_sessions/all_times", [sesss], [rat];
        break_session=true);

    if (region == "STR") || (region == "FOF")
        delay = 0.05
    elseif region == "PPC"
        delay = 0.1
    end
    
    use_bin_center = true;
    #data = map(x->bin_clicks_spikes_and_λ0!(x, use_bin_center; dt=dt,delay=delay), data);
    data = map(x-> map(y-> bin_clicks_spikes_and_λ0!(y,use_bin_center; dt=dt,delay=delay), x), data);
    data = vcat(data...)

    nsessions, N_per_sess, dimy, f_str = length(data), map(data-> data["N"], data), 4, "sig"
    
    if isfile(path*"/data/results/working/by_rat/softplus_sig/break/"*region*"_"*rat*".jld")
        JLD2.@load path*"/data/results/working/by_rat/softplus_sig/break/"*region*"_"*rat*".jld" pz py
        
    else
        
        pz = Dict("name" => ["σ_i","B", "λ", "σ_a","σ_s","ϕ","τ_ϕ"],
            "fit" => vcat(falses(1),trues(2),falses(4)),
            "initial" => [1., 10., -0.1, 2*eps(), 2*eps(), 1.0-eps(), 0.005],
            "lb" => [eps(), 8., -5., eps(), eps(), eps(), eps()],
            "ub" => [10., 40, 5., 100., 2.5, 2., 1.])

        py = Dict("fit" => map(N-> repeat([trues(dimy)],outer=N), N_per_sess),
            "initial" => [[[Vector{Float64}(undef,dimy)] for n in 1:N] for N in N_per_sess],
            "dimy"=> dimy,
            "N"=> N_per_sess,
            "nsessions"=> nsessions)

        py["initial"] = map(data-> regress_init(data, f_str), data)
        #pz["fit"] =  vcat(falses(1),trues(2),falses(4))
        pz, py = optimize_model(pz, py, data, f_str, show_trace=true, iterations=200)

        pz["initial"] = vcat(1.,10.,-0.1,20.,0.5,1.0-eps(),0.005)
        pz["state"][pz["fit"] .== false] = pz["initial"][pz["fit"] .== false]
        pz["fit"] = vcat(trues(7))
        
    end

    @time pz, py, = optimize_model(pz, py, data, f_str, n, show_trace=true, iterations=500) 
    @time pz, py, H = compute_H_CI!(pz, py, data, f_str, n)

    LL_ML = compute_LL(pz["final"], py["final"], data, n, f_str)

    LL_null = mapreduce(d-> mapreduce(r-> mapreduce(n-> 
                neural_null(d["spike_counts"][r][n], d["λ0"][r][n], d["dt"]), 
                    +, 1:d["N"]), +, 1:d["ntrials"]), +, data)

    ΔLL = LL_ML - LL_null
    
    JLD2.@save path*"/data/results/working/by_rat/softplus_sig/break/"*region*"_"*rat*".jld" pz py ΔLL H data

    try

        μ_hat_ct = pulse_input_DDM.sample_average_expected_rates_multiple_sessions(pz["final"], 
            py["final"], data, f_str)

        JLD2.@save path*"/data/results/working/by_rat/softplus_sig/break/"*region*"_"*rat*".jld" pz py ΔLL μ_hat_ct H data
        #BSON.@save path*"/data/results/working/by_rat/softplus_sig/w_adapt/"*region*"_"*rat*".bson" pz py ΔLL μ_hat_ct data
        
    catch
        nothing
    end
    
    
elseif model == "choice"
    
    use_bin_center = true;
    data = aggregate_choice_data(path*"/data/hanks_data_sessions/all_times",[sesss],[rat])
    data = bin_clicks!(data,use_bin_center;dt=dt)
    
    if isfile(path*"/data/results/working/by_rat/choice/"*region*"_"*rat*".jld")
        JLD2.@load path*"/data/results/working/by_rat/choice/"*region*"_"*rat*".jld" pz pd
        
    else
        
        #parameters for the choice observation
        pd = Dict("name" => vcat("bias","lapse"), "fit" => trues(2), 
            "initial" => vcat(0.,0.5))

        pz = Dict("name" => ["σ_i","B", "λ", "σ_a","σ_s","ϕ","τ_ϕ"],
                "fit" => vcat(trues(7)),
                "initial" => [1., 10., -0.1, 20.,0.5, 1.0-eps(), 0.005],
                "lb" => [eps(), 8., -5., eps(), eps(), eps(), eps()],
                "ub" => [10., 40, 5., 100., 2.5, 2., 1.])
        
    end
        
    @time pz, pd = optimize_model(pz, pd, data; n=n, show_trace=true, iterations=500)
    @time pz, pd, H = compute_H_CI!(pz, pd, data; n=n)
        
    LL_ML = compute_LL(pz["final"], pd["final"], data; n=n)
    
    LL_null = choice_null(data["pokedR"])
    
    ΔLL = LL_ML - LL_null
    
    JLD2.@save path*"/data/results/working/by_rat/choice/"*region*"_"*rat*".jld" pz pd ΔLL H
    
    try
        choices_hat = sample_choices_all_trials(data, pz["final"], pd["final"])   
        JLD2.@save path*"/data/results/working/by_rat/choice/"*region*"_"*rat*".jld" pz pd ΔLL choices_hat H
        #BSON.@save path*"/data/results/working/by_rat/choice/"*region*"_"*rat*".bson" pz pd ΔLL choices_hat data
    catch
        nothing
    end
    
end