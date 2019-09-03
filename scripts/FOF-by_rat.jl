using Revise, Distributed

addprocs(44);
@everywhere using Pkg; @everywhere Pkg.activate("/usr/people/briandd/Projects/pulse_input_DDM.jl")

@everywhere using pulse_input_DDM
using JLD2, PyPlot, Statistics

path = ENV["HOME"]*"/Projects/pulse_input_DDM.jl"
f_str, n, dt = "sig", 103, 1e-2;

#region = "FOF"
region = ARGS[1]

sessids, ratnames = sessids_from_region(region);

idx = parse(Int,ARGS[2])

#rat = ratnames[3];
#sesss = sessids[3];

rat = ratnames[idx]
sesss = sessids[idx]

data = aggregate_spiking_data(path*"/data/hanks_data_sessions/all_times", [sesss], [rat]);

if (region == "STR") || (region == "FOF")
    delay = 0.05
elseif region == "PPC"
    delay = 0.1
end

use_bin_center = true;
data = map(x->bin_clicks_spikes_and_λ0!(x, use_bin_center; dt=dt,delay=delay), data);

pz = Dict("name" => ["σ_i","B", "λ", "σ_a","σ_s","ϕ","τ_ϕ"],
            "fit" => vcat(falses(1),trues(2),falses(4)),
            "initial" => [1., 10., -0.1, 2*eps(), 2*eps(), 1.0-eps(), 0.005],
            "lb" => [eps(), 8., -5., eps(), eps(), eps(), eps()],
            "ub" => [10., 40, 5., 100., 2.5, 2., 1.])

nsessions, N_per_sess, dimy = length(data), map(data-> data["N"], data), 4

py = Dict("fit" => map(N-> repeat([trues(dimy)],outer=N), N_per_sess),
    "initial" => [[[Vector{Float64}(undef,dimy)] for n in 1:N] for N in N_per_sess],
    "dimy"=> dimy,
    "N"=> N_per_sess,
    "nsessions"=> nsessions)

py["initial"] = map(data-> regress_init(data, f_str), data)
pz, py = optimize_model(pz, py, data, f_str, show_trace=true, iterations=200)

pz["initial"] = vcat(1.,10.,-0.1,20.,0.5,1.0-eps(),0.005)
pz["state"][pz["fit"] .== false] = pz["initial"][pz["fit"] .== false]
pz["fit"] = vcat(trues(7))

@time pz, py, = optimize_model(pz, py, data, f_str, n, show_trace=true, iterations=500) 
@time pz, py = compute_H_CI!(pz, py, data, f_str, n)

LL_ML = compute_LL(pz["final"], py["final"], data, n, f_str)

LL_null = mapreduce(d-> mapreduce(r-> mapreduce(n-> 
            neural_null(d["spike_counts"][r][n], d["λ0"][r][n], d["dt"]), 
                +, 1:d["N"]), +, 1:d["ntrials"]), +, data)

ΔLL = LL_ML - LL_null

μ_hat_ct = pulse_input_DDM.sample_average_expected_rates_multiple_sessions(pz["final"], 
    py["final"], data, f_str)

#=

num_rows, num_cols = length(data), maximum(map(x-> x["N"], data))
fig, ax = subplots(num_rows, num_cols, figsize=(4*maximum(map(x-> x["N"], data)),4*length(data)))
my_colors = ["#E50000","#9F3F00","#5A7F00","#15BF00"]
#PPC colors = ["#1822A0","#5D4A7A","#A37354","#E99C2F"]
#STR colors = ["#A01892","#B85C71","#D0A150","#E9E62F"]

for i in 1:num_rows
    
    plt_data = data[i]
    plt_data_2 = μ_hat_ct[i]
    
    for j in 1:length(plt_data_2)               
        for k = 1:plt_data["nconds"]        
            
            if num_cols > 1
            
                ax[i,j].fill_between((1:length(plt_data["μ_ct"][j][k]))*dt,
                    plt_data["μ_ct"][j][k] + plt_data["σ_ct"][j][k],
                    plt_data["μ_ct"][j][k] - plt_data["σ_ct"][j][k],
                    alpha=0.2, color=my_colors[k])
                
                ax[i,j].plot((1:length(plt_data_2[j][k]))*dt,
                    plt_data_2[j][k], color=my_colors[k])                                     

                ax[i,j].set_xlim((0, 0.5))
                #ax[j].set_ylim((0, 30))

                
            else
                
                ax.fill_between((1:length(plt_data["μ_ct"][j][k]))*dt,
                    plt_data["μ_ct"][j][k] + plt_data["σ_ct"][j][k],
                    plt_data["μ_ct"][j][k] - plt_data["σ_ct"][j][k],
                    alpha=0.2, color=my_colors[k])
                
                ax.plot((1:length(plt_data_2[j][k]))*dt,
                    plt_data_2[j][k], color=my_colors[k])
            
                ax.set_xlim((0, 0.5))
                ax.set_ylim((0, 30))
                
            end
                        
        end   
        
        if num_cols > 1
            ax[i,j].plot((1:length(plt_data["μ_t"][j]))*dt,
                plt_data["μ_t"][j], color="black")
        else
            ax.plot((1:length(plt_data["μ_t"][j]))*dt,
                plt_data["μ_t"][j], color="black")
        end
                      
    end
    
end

tight_layout() 

=#

JLD2.@save path*"/data/results/working/by_rat/softplus_sig/w_adapt/"*region*"_"*rat*".jld" pz py ΔLL μ_hat_ct
