"""
"""
function boot_LL(pz,py,data,f_str,i,n)
    dcopy = deepcopy(data)
    dcopy["spike_counts"] = sample_spikes_multiple_sessions(pz, py, [dcopy], f_str; rng=i)[1]    
    
    LL_ML = compute_LL(pz, py, [dcopy], n, f_str)

    #LL_null = mapreduce(d-> mapreduce(r-> mapreduce(n-> 
    #            neural_null(d["spike_counts"][r][n], d["λ0"][r][n], d["dt"]), 
    #                +, 1:d["N"]), +, 1:d["ntrials"]), +, [data])

    #(LL_ML - LL_null) / dcopy["ntrials"]    

    LL_null = mapreduce(d-> mapreduce(r-> mapreduce(n-> 
        neural_null(d["spike_counts"][r][n], map(λ-> f_py(0.,λ, py[1][n],f_str), d["λ0"][r][n]), d["dt"]), 
            +, 1:d["N"]), +, 1:d["ntrials"]), +, [dcopy])

    #return 1. - (LL_ML/LL_null), LL_ML, LL_null
    LL_ML - LL_null
    
end


"""
"""
function sample_clicks_and_spikes(pz::Vector{Float64}, py::Vector{Vector{Vector{Float64}}}, 
        f_str::String, num_sessions::Int, num_trials_per_session::Vector{Int}; use_bin_center::Bool=false,
        dtMC::Float64=1e-4, rng::Int=0)
        
    data = map((ntrials,rng)-> sample_clicks(ntrials; rng=rng), num_trials_per_session, (1:num_sessions) .+ rng) 
      
    map((data,py) -> data=sample_λ0!(data, py; dtMC=dtMC), data, py)
    
    Y = sample_spikes_multiple_sessions(pz, py, data, f_str, use_bin_center, dtMC; rng=rng)      
    map((data,Y)-> data["spike_counts"] = Y, data, Y) 
        
    return data
    
end

function sample_λ0!(data, py::Vector{Vector{Float64}}; dtMC::Float64=1e-4, rng::Int=1)
    
    data["dt_synthetic"], data["synthetic"], data["N"] = dtMC, true, length(py)
            
    #Random.seed!(rng)   
    #data["λ0"] = [repeat([collect(range(10. *rand(),stop=10. * rand(), 
    #                    length=Int(ceil(T./dt))))], outer=length(py)) for T in data["T"]]
    data["λ0"] = [repeat([zeros(Int(ceil(T./dtMC)))], outer=length(py)) for T in data["T"]]
            
    return data
    
end


"""
"""
function sample_spikes_multiple_sessions(pz::Vector{Float64}, py::Vector{Vector{Vector{Float64}}}, 
        data, f_str::String, use_bin_center::Bool, dt::Float64; rng::Int=1)
    
    λ, = sample_expected_rates_multiple_sessions(pz, py, data, f_str, use_bin_center, dt; rng=rng) 
    Y = map((λ,data)-> map(λ-> map(λ-> poisson_noise!.(λ, dt), λ), λ), λ, data)         
    #Y = map((py,λ0)-> poisson_noise!.(map((a, λ0)-> f_py!(a, λ0, py, f_str), a, λ0), dt), py, λ0)  
            
    #this assumes only one spike per bin, which should most often be true at 1e-4, but not guaranteed!
    #findall(x-> x > 1, pulse_input_DDM.poisson_noise!.(10 * ones(100 * 10 * Int(1. /1e-4)),1e-4))
    #Y = map((py,λ0)-> findall(x -> x != 0, 
    #        poisson_noise!.(map((a, λ0)-> f_py!(a, λ0, py, f_str=f_str), a, λ0), dt)) .* dt, py, λ0)   
    
    return Y
    
end

function sample_expected_rates_multiple_sessions(pz::Vector{Float64}, py::Vector{Vector{Vector{Float64}}}, 
        data, f_str::String, use_bin_center::Bool, dt::Float64; rng::Int=1)
    
    nsessions = length(data)
      
    output = map((data, py)-> sample_expected_rates_single_session(data, pz, py, f_str, use_bin_center, dt; rng=rng), 
        data, py)   
    
    λ = map(x-> x[1], output)
    a = map(x-> x[2], output)  
    
    return λ, a
    
end

function sample_expected_rates_single_session(data::Dict, pz::Vector{Float64}, py::Vector{Vector{Float64}}, 
        f_str::String, use_bin_center::Bool, dt::Float64; rng::Int=1)
    
    Random.seed!(rng)   
    
    T, L, R, λ0 = data["T"], data["leftbups"], data["rightbups"], data["λ0"]
    nT, nL, nR = bin_clicks(T,L,R;dt=dt, use_bin_center=use_bin_center)
    
    output = pmap((λ0,nT,L,R,nL,nR,rng) -> sample_expected_rates_single_trial(pz,py,λ0,nT,L,R,nL,nR,
        f_str,use_bin_center,dt; rng=rng), λ0, nT, L, R, nL, nR, shuffle(1:length(T)))    
    
    λ = map(x-> x[1], output)
    a = map(x-> x[2], output)  
    
    return λ,a
    
end

function sample_expected_rates_single_trial(pz::Vector{Float64}, py::Vector{Vector{Float64}}, λ0::Vector{Vector{Float64}}, 
        nT::Int, L::Vector{Float64}, R::Vector{Float64}, nL::Vector{Int}, nR::Vector{Int},
        f_str::String, use_bin_center::Bool, dt::Float64; rng::Int=1)
    
    Random.seed!(rng)  
    a = sample_latent(nT,L,R,nL,nR,pz,use_bin_center;dt=dt)
    λ = map((py,λ0)-> map((a, λ0)-> f_py!(a, λ0, py, f_str), a, λ0), py, λ0)  
    
    return λ, a
    
end

poisson_noise!(lambda,dt) = Int(rand(Poisson(lambda*dt)))


"""
"""
function sample_per_trial_expected_rates_multiple_sessions(pz, py, data, f_str::String)
    
    output = map(i-> sample_expected_rates_multiple_sessions(pz, py, data, f_str; rng=i), 1:100)
    μ_rate = mean(map(k-> output[k][1], 1:length(output)))
    
    return μ_rate
    
end


"""
"""
function sample_average_expected_rates_multiple_sessions(pz, py, data, f_str::String)
    
    output = map(i-> sample_expected_rates_multiple_sessions(pz, py, data, f_str; rng=i), 1:100)
    μ_rate = mean(map(k-> output[k][1], 1:length(output)))
    
    #μ_hat_ct = map(i-> map(n-> map(c-> [mean([μ_rate[i][data[i]["conds"] .== c][k][n][t] 
    #    for k in findall(data[i]["nT"][data[i]["conds"] .== c] .>= t)]) 
    #    for t in 1:(maximum(data[i]["nT"][data[i]["conds"] .== c]))], 
    #            1:data[i]["nconds"]), 
    #                1:data[i]["N"]), 
    #                    1:length(data))
    
    μ_hat_ct = map(i-> condition_mean_varying_duration_trials(μ_rate[i], data[i]["conds"], 
            data[i]["nconds"], data[i]["N"], data[i]["nT"]), 1:length(data))
    
    return μ_hat_ct
    
end


"""
"""
function condition_mean_varying_duration_trials(μ_rate, conds, nconds, N, nT)
    
    map(n-> map(c-> [mean([μ_rate[conds .== c][k][n][t] 
        for k in findall(nT[conds .== c] .>= t)]) 
        for t in 1:(maximum(nT[conds .== c]))], 
                1:nconds), 1:N)
    
end