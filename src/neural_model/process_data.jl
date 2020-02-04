"""
    save(file, model, options, CI)

Given a file, model produced by optimize and options, save the results of the optimization to a .MAT file
"""
function save(file, model::neuralDDM, options, CI)

    @unpack lb, ub, fit = options
    @unpack θ = model
    
    dict = Dict("ML_params"=> collect(pulse_input_DDM.flatten(θ)),
        "lb"=> lb, "ub"=> ub, "fit"=> fit,
        "CI" => CI)

    matwrite(file, dict)

    #=
    if !isempty(H)
        #dict["H"] = H
        hfile = matopen(path*"hessian_"*file, "w")
        write(hfile, "H", H)
        close(hfile)
    end
    =#

end



"""
"""
function train_test_divide(data,frac)

    train_trials = StatsBase.sample(1:data["ntrials"], Int(ceil(0.8 * data["ntrials"])), replace = false)
    test_trials = setdiff(1:data["ntrials"], train_trials)

    train_data, test_data = deepcopy(data), deepcopy(data)

    for key in collect(keys(data))
        if length(data[key]) == data["ntrials"]
            train_data[key] = data[key][train_trials]
            test_data[key] = data[key][test_trials]
        end
    end

    #all of the mus and sigams should be filtered too, but not important for fitting.
    train_data["ntrials"], test_data["ntrials"] = length(train_trials), length(test_trials)

    return train_data, test_data

end


"""
    load_neural_data(path, files)

Load neural data .MAT files and return a Dict.
"""
function load(file::String, break_sim_data::Bool, centered::Bool=true;
        dt::Float64=1e-2, delay::Float64=0., pad::Int=10, filtSD::Int=5,
        cut::Int=10)

    data = read(matopen(file), "rawdata")

    T = vec(data["T"])
    L = vec(map(x-> vec(collect(x)), data[collect(keys(data))[occursin.("left", collect(keys(data)))][1]]))
    R = vec(map(x-> vec(collect(x)), data[collect(keys(data))[occursin.("right", collect(keys(data)))][1]]))
    choices = vec(convert(BitArray, data["pokedR"]))

    click_times = clicks.(L, R, T)
    binned_clicks = bin_clicks(click_times, centered=centered, dt=dt)

    ncells = size(data["spike_times"][1], 2)

    #=
    if break_sim_data

        rawdata["N"] = 1

        for n = 1:ncells

            subdata = deepcopy(rawdata)
            subdata["spike_times"] = vec(map(x-> vec(collect([x[n]])), rawdata["spike_times"]))
            push!(data, subdata)

        end

    else
    =#

    spikes = vec(map(x-> vec(vec.(collect.(x))), data["spike_times"]))

    #end

    nT = map(x-> x.nT, binned_clicks)

    output = map((spikes, nT)-> bin_spikes(spikes, dt, nT; delay=delay, pad=pad), spikes, nT)
    spikes = getindex.(output, 1)
    padded = getindex.(output, 2)
    
    μ_rnt = filtered_rate.(padded, dt; filtSD=filtSD, cut=cut)
        
    μ_t = map(n-> [max(0., mean([μ_rnt[i][n][t]
        for i in findall(nT .+ (pad - cut) .>= t)]))
        for t in 1:(maximum(nT) .+ (pad - cut))], 1:ncells)
    
    λ0 = map(nT-> bin_λ0(μ_t, nT), nT)
    #λ0 = map(nT-> map(μ_t-> zeros(nT), μ_t), nT)

    input_data = neuralinputs(click_times, binned_clicks, λ0, dt, centered)
    
    return neuraldata(input_data, spikes, ncells, choices), μ_rnt, μ_t
    
end


"""
"""
#function bin_clicks_spikes_λ0(data::Dict; centered::Bool=true,
#        dt::Float64=1e-2, delay::Float64=0., pad::Int=10, filtSD::Int=5)

function bin_clicks_spikes_λ0(spikes, λ0, clicks; centered::Bool=true,
        dt::Float64=1e-2, delay::Float64=0., dt_synthetic::Float64=1e-4,
        synthetic::Bool=false)

    spikes = bin_spikes(spikes, dt, dt_synthetic)
    λ0 = bin_λ0(λ0, dt, dt_synthetic)
    binned_clicks = bin_clicks(clicks, centered=centered, dt=dt)

    return spikes, λ0, binned_clicks

    #T, L, R = data["T"], data["leftbups"], data["rightbups"]
    #binned_clicks = bin_clicks(clicks(L, R, T, data["ntrials"]), centered=centered, dt=dt)
    #@unpack nT, nL, nR, dt, centered = binned_clicks
    #data["nT"] = nT
    #data["binned_leftbups"] = nL
    #data["binned_rightbups"] = nR
    #data["dt"] = dt
    #data["use_bin_center"] = centered

    #data["spike_counts"] = bin_spikes(data["spike_counts"], dt; delay=delay, synthetic=true)
    #data = pad_binned_spikes!(data; delay=delay, pad=pad)
    #data = compute_filtered_rate!(data; filtSD=filtSD)
    #data["λ0"] = compute_λ0(data["λ0"], dt; synthetic=true)
    #data = compute_λ0!(data)

    #return data

end


"""
"""
bin_λ0(λ0::Vector{Vector{Vector{Float64}}}, dt, dt_synthetic) = bin_λ0.(λ0, dt, dt_synthetic)


"""
"""
bin_λ0(λ0::Vector{Vector{Float64}}, dt, dt_synthetic) = decimate.(λ0, Int(dt/dt_synthetic))


"""
"""
bin_spikes(spikes::Vector{Vector{Vector{Int}}}, dt, dt_synthetic) = bin_spikes.(spikes, dt, dt_synthetic)



"""
"""
bin_spikes(spikes::Vector{Vector{Int}}, dt::Float64, dt_synthetic::Float64) = 
    map(SCn-> sum.(Iterators.partition(SCn, Int(dt/dt_synthetic))), spikes)


"""
"""
function bin_spikes(spike_times::Vector{Vector{Float64}}, dt, nT::Int; delay::Float64=0., pad::Int=10) 

    trial = map(x-> fit(Histogram, vec(collect(x .- delay)), 
            collect(range(0, stop=nT*dt, length=nT+1)), closed=:left).weights, spike_times)

    padded = map(x-> fit(Histogram, vec(collect(x .- delay)), 
            collect(range(-pad*dt, stop=(nT+pad)*dt, length=(nT+2*pad)+1)), closed=:left).weights, spike_times)


    return trial, padded
    
end


"""
"""
bin_λ0(λ0::Vector{Vector{Float64}}, nT) = map(λ0-> λ0[1:nT], λ0)


"""
"""
function filtered_rate(padded, dt; filtSD::Int=5, cut::Int=10)

    kern = reflect(KernelFactors.gaussian(filtSD, 8*filtSD+1));

    map(padded-> imfilter(1/dt * padded, kern,
            Fill(zero(eltype(padded))))[cut+1:end-cut], padded)

end


"""
"""
function process_spike_data(μ_rnt, data, ncells; pad::Int=10, cut::Int=10, nconds::Int=4)

    nT = map(x-> x.input_data.binned_clicks.nT, data)
    μ_rn = map(n-> map(μ_rnt-> mean(μ_rnt[n]), μ_rnt), 1:ncells)

    #if haskey(data,"pokedR")
    #    data["d'"] = map(n-> dprime(data["μ_rn"][n], data["pokedR"]), 1:data["N"])
    #end

    ΔLRT = last.(diffLR.(data))
    #data["nconds"] = nconds
    conds = encode(LinearDiscretizer(binedges(DiscretizeUniformWidth(nconds), ΔLRT)), ΔLRT)

    μ_ct = map(n-> map(c-> [mean([μ_rnt[conds .== c][i][n][t]
        for i in findall(nT[conds .== c] .+ (pad - cut) .>= t)])
        for t in 1:(maximum(nT[conds .== c]) .+ (pad - cut))], 1:nconds), 1:ncells)

    σ_ct = map(n-> map(c-> [std([μ_rnt[conds .== c][i][n][t]
        for i in findall(nT[conds .== c] .+ (pad - cut) .>= t)]) /
            sqrt(length([μ_rnt[conds .== c][i][n][t]
                for i in findall(nT[conds .== c] .+ (pad - cut) .>= t)]))
        for t in 1:(maximum(nT[conds .== c]) .+ (pad - cut))], 1:nconds), 1:ncells)

    return μ_ct, σ_ct, μ_rn

end

group_by_neuron(data) = [[data[t].spikes[n] for t in 1:length(data)] for n in 1:data[1].ncells]

function filter_data_by_dprime!(data,thresh)

    bool_vec = data["d'"] .< thresh
    deleteat!(data["μ_t"], bool_vec)
    deleteat!(data["μ_ct"], bool_vec)
    deleteat!(data["σ_ct"], bool_vec)
    deleteat!(data["μ_rn"], bool_vec)
    map(x-> deleteat!(x, bool_vec), data["spike_counts_padded"])
    map(x-> deleteat!(x, bool_vec), data["spike_counts"])
    map(x-> deleteat!(x, bool_vec), data["spike_times"])
    map(x-> deleteat!(x, bool_vec), data["μ_rnt"])
    map(x-> deleteat!(x, bool_vec), data["λ0"])
    map(x-> deleteat!(x, bool_vec), data["cellID"])

    deleteat!(data["d'"], bool_vec);
    data["N"] = length(data["d'"])

    return data

end

function filter_data_by_ΔLL!(data,ΔLL,thresh)

    bool_vec = ΔLL .< thresh
    deleteat!(data["μ_t"], bool_vec)
    deleteat!(data["μ_ct"], bool_vec)
    deleteat!(data["σ_ct"], bool_vec)
    deleteat!(data["μ_rn"], bool_vec)
    map(x-> deleteat!(x, bool_vec), data["spike_counts_padded"])
    map(x-> deleteat!(x, bool_vec), data["spike_counts"])
    map(x-> deleteat!(x, bool_vec), data["spike_times"])
    map(x-> deleteat!(x, bool_vec), data["μ_rnt"])
    map(x-> deleteat!(x, bool_vec), data["λ0"])
    map(x-> deleteat!(x, bool_vec), data["cellID"])

    deleteat!(data["d'"], bool_vec);
    data["N"] = length(data["d'"])

    return data

end

#################################### Data filtering #########################

#This doesn't work, but might be a good idea to fix up.

function filter_data_by_cell!(data,cell_index)

    data["N0"] = length(cell_index)

    #organized by neurons, so filter by neurons
    data["spike_counts_by_neuron"] = data["spike_counts_by_neuron"][cell_index]
    data["trial"] = data["trial"][cell_index]
    data["cellID_by_neuron"] = data["cellID_by_neuron"][cell_index]
    data["sessID_by_neuron"] = data["sessID_by_neuron"][cell_index]
    data["ratID_by_neuron"] = data["ratID_by_neuron"][cell_index]

    if (haskey(data,"spike_counts_stimulus_aligned_extended_by_neuron") |
        haskey(data, "spike_counts_cpoke_aligned_extended_by_neuron"))

        data["spike_counts_stimulus_aligned_extended_by_neuron"] =
            data["spike_counts_stimulus_aligned_extended_by_neuron"][cell_index]
        data["spike_counts_cpoke_aligned_extended_by_neuron"] =
            data["spike_counts_cpoke_aligned_extended_by_neuron"][cell_index]

    end

    trial_index = unique(collect(vcat(data["trial"]...)))
    data["trial0"] = length(trial_index)

    #organized by trials, so filter by trials
    data["binned_leftbups"] = data["binned_leftbups"][trial_index]
    data["binned_rightbups"] = data["binned_rightbups"][trial_index]
    data["rightbups"] = data["rightbups"][trial_index]
    data["leftbups"] = data["leftbups"][trial_index]
    data["T"] = data["T"][trial_index]
    data["nT"] = data["nT"][trial_index]
    data["pokedR"] = data["pokedR"][trial_index]
    data["correct_dir"] = data["correct_dir"][trial_index]
    data["sessID"] = data["sessID"][trial_index]
    data["ratID"] = data["ratID"][trial_index]
    data["stim_start"] = data["stim_start"][trial_index]
    data["cpoke_end"] = data["cpoke_end"][trial_index]

    #this subtracts the minimum current trial index from all of the trial indices
    #for i = 1:data["N0"]
    #    #data["trial"] = map(x->x[1] - minimum(trial_index) + 1 : x[end] - minimum(trial_index) + 1, data["trial"])
    #    data["trial"][i] = data["trial"][i][1] - minimum(trial_index) + 1 : data["trial"][i][end] - minimum(trial_index) + 1
    #end

    #tvec2 = deepcopy(unique(vcat(data["trial"]...)))
    #map!(x->findall(x[1] .== tvec2)[1]:findall(x[end] .== tvec2)[1], data["trial"], data["trial"])

    #trial_index = unique(collect(vcat(data["trial"]...)))

    #shifts all trial times so the run consequtively from 1:data["trial0"]
    #for i = 1:data["N0"]
    #    data["trial"][i] = findfirst(data["trial"][i][1] .== trial_index) : findfirst(data["trial"][i][end] .== trial_index)
    #end

    data["N"] = Vector{Vector{Int}}(undef,0)
    map(x->push!(data["N"], Vector{Int}(undef,0)), 1:data["trial0"])
    data["cellID"] = Vector{Vector{Int}}(undef,0)
    map(x->push!(data["cellID"], Vector{Int}(undef,0)), 1:data["trial0"])
    data["spike_counts"] = Vector{Vector{Vector{Int64}}}(undef,0)
    map(x->push!(data["spike_counts"], Vector{Vector{Int}}(undef,0)), 1:data["trial0"])

    #map(y->map(x->push!(data["N"][x],y), data["trial"][y]), 1:data["N0"])
    #map(y->map(x->push!(data["cellID"][x], cell_index[y]), data["trial"][y]), 1:data["N0"])
    #map(y->map(x->push!(data["spike_counts"][data["trial"][y][x]], data["spike_counts_by_neuron"][y][x]),
    #        1:length(data["trial"][y])), 1:data["N0"])

    for i = 1:data["N0"]

        #shifts all trial times so the run consequtively from 1:data["trial0"]
        data["trial"][i] = findfirst(data["trial"][i][1] .== trial_index) : findfirst(data["trial"][i][end] .== trial_index)

        for j = 1:length(data["trial"][i])

            push!(data["N"][data["trial"][i][j]], i)
            push!(data["cellID"][data["trial"][i][j]], data["cellID_by_neuron"][i])
            push!(data["spike_counts"][data["trial"][i][j]], data["spike_counts_by_neuron"][i][j])

        end
    end

    return data

end

#=

######INCOMPLETE##########

function aggregate_and_append_extended_spiking_data!(data::Dict, path::String, sessids::Vector{Vector{Int}},
        ratnames::Vector{String}, dt::Float64, ts::Float64, tf::Float64; delay::Float64=0.)

    data["spike_counts_stimulus_aligned_extended"] = Vector{Vector{Vector{Int64}}}()
    data["spike_counts_cpoke_aligned_extended"] = Vector{Vector{Vector{Int64}}}()
    data["spike_counts_stimulus_aligned_extended_by_neuron"] = Vector{Vector{Vector{Int64}}}()
    data["spike_counts_cpoke_aligned_extended_by_neuron"] = Vector{Vector{Vector{Int64}}}()
    map(x-> push!(data["spike_counts_stimulus_aligned_extended_by_neuron"], Vector{Vector{Int}}(undef,0)), 1:data["N0"])
    map(x-> push!(data["spike_counts_cpoke_aligned_extended_by_neuron"], Vector{Vector{Int}}(undef,0)), 1:data["N0"])
    data["time_basis_edges"] = (floor.(ts/dt) * dt) : dt : (ceil.(tf/dt) * dt)
    #data["time_basis_centers"] = (data["time_basis_edges"][1] + dt/2): dt: (data["time_basis_edges"][end-1] + dt/2)
    data["time_basis_centers"] = data["time_basis_edges"][1:end-1]

    for j = 1:length(ratnames)
        for i = 1:length(sessids[j])
            rawdata = read(matopen(path*"/"*ratnames[j]*"_"*string(sessids[j][i])*".mat"),"rawdata")
            data = append_extended_neural_data!(data, rawdata, dt, ts, tf, delay=delay)
        end
    end

    map(n-> map(t-> append!(data["spike_counts_stimulus_aligned_extended_by_neuron"][n],
        data["spike_counts_stimulus_aligned_extended"][t][data["N"][t] .== n]), data["trial"][n]), 1:data["N0"])

    map(n-> map(t-> append!(data["spike_counts_cpoke_aligned_extended_by_neuron"][n],
        data["spike_counts_cpoke_aligned_extended"][t][data["N"][t] .== n]), data["trial"][n]), 1:data["N0"])

    return data

end

function append_extended_neural_data!(data::Dict, rawdata::Dict, dt::Float64, ts::Float64, tf::Float64;
        delay::Float64=0.)

    N = size(rawdata["spike_times"][1],2)
    ntrials = length(rawdata["T"])

    #time = data["time_basis_edges"]

    binnedT = ceil.(Int,rawdata["T"]/dt)

    append!(data["spike_counts_stimulus_aligned_extended"], map((t,tri) -> map(n -> fit(Histogram, vec(collect(tri[n] .- delay)),
        -10*dt:dt:((t+10)*dt), closed=:left).weights, 1:N), binnedT, rawdata["spike_times"]))

    #append!(data["spike_counts_cpoke_aligned_extended"], map(tri -> map(n -> fit(Histogram, vec(collect(tri[n])),
    #    time, closed=:left).weights, 1:N), rawdata["spike_times"]))

    time = data["time_basis_edges"]

    for i = 1:ntrials

        blah = map(n -> fit(Histogram, vec(collect(rawdata["spike_times"][i][n] .+ rawdata["stim_start"][i] .- delay)),
                time, closed=:left).weights, 1:N)
        push!(data["spike_counts_cpoke_aligned_extended"], blah)

    end

    return data

end

function λ0_by_trial(data::Dict, μ_λ; cpoke_aligned::Bool=false,
        extended::Bool=false)

    λ0 = Dict("by_trial" => Vector{Vector{Vector{Float64}}}(undef,0),
        "by_neuron" => Vector{Vector{Vector{Float64}}}())

    map(x->push!(λ0["by_trial"] , Vector{Vector{Float64}}(undef,0)), 1:data["trial0"])
    map(x-> push!(λ0["by_neuron"], Vector{Vector{Float64}}(undef,0)), 1:data["N0"])

    for i = 1:data["N0"]
        for j = 1:length(data["trial"][i])

            if extended

                if cpoke_aligned
                    stim_start = data["stim_start"][data["trial"][i][j]]
                else
                    stim_start =  0.
                end
                T0 = findlast(collect(data["time_basis_edges"]) .<= stim_start)

            else
                T0 = 1
            end

            nT = data["nT"][data["trial"][i][j]]

            push!(λ0["by_trial"][data["trial"][i][j]], μ_λ[i][T0: T0+ nT - 1])

        end
    end

    map(n-> map(t-> append!(λ0["by_neuron"][n],
                λ0["by_trial"][t][data["N"][t] .== n]), data["trial"][n]), 1:data["N0"])

    return λ0

end

function append_neural_data!(data::Dict, rawdata::Dict, ratname::String, sessID::Int, dt::Float64;
        delay::Float64=0.)

    N = size(rawdata["spike_times"][1],2)
    ntrials = length(rawdata["T"])

    #by trial
    #append!(data["cellID"], map(x-> vec(collect(x)), rawdata["cellID"]))
    #append!(data["stim_start"], rawdata["stim_start"])

    #by neuron
    append!(data["cellID_by_neuron"], rawdata["cellID"][1])
    append!(data["sessID_by_neuron"], repeat([sessID], inner=N))
    append!(data["ratID_by_neuron"], repeat([ratname], inner=N))
    append!(data["trial"], repeat([data["trial0"]+1 : data["trial0"]+ntrials], inner=N))

    #if organize == "by_trial"

    #by trial
    binnedT = ceil.(Int,rawdata["T"]/dt)

    append!(data["cellID"], map(x-> vec(collect(x)), rawdata["cellID"]))

    append!(data["spike_counts"], map((x,y) -> map(z -> fit(Histogram, vec(collect(y[z] .- delay)),
            0.:dt:x*dt, closed=:left).weights, 1:N), binnedT, rawdata["spike_times"]))
    append!(data["N"], repeat([collect(data["N0"]+1 : data["N0"]+N)], inner=ntrials))

    #by neuron
    #append!(data["trial"], repeat([data["trial0"]+1 : data["trial0"]+ntrials], inner=N))

    #elseif organize == "by_neuron"

    #    append!(data["spike_counts"],map!(z -> map!((x,y) -> fit(Histogram,vec(collect(y[z])),
    #            0.:dt:x*dt,closed=:left).weights, Vector{Vector}(undef,ntrials),
    #            binnedT,rawdata["spike_times"]), Vector{Vector}(undef,N),1:N))

    #    append!(data["spike_counts_all"], map!(z -> map!((x,y) -> fit(Histogram,vec(collect(y[z])),
    #            0.:dt:x*dt, closed=:left).weights, Vector{Vector}(undef,ntrials),
    #            binnedT, rawdata["spike_times"]), Vector{Vector}(undef,N), 1:N))

    #    append!(data["trial"],repeat([data["trial0"]+1:data["trial0"]+ntrials],inner=N))

    #end

    data["N0"] += N
    data["trial0"] += ntrials

    return data

end

#function group_by_neuron!(data)

    #trials = Vector{Vector{Int}}()
    #data["spike_counts_by_neuron"] = Vector{Vector{Vector{Int64}}}()

    #map(x->push!(trials,Vector{Int}(undef,0)),1:data["N0"])
    #map(x-> push!(data["spike_counts_by_neuron"], Vector{Vector{Int}}(undef,0)), 1:data["N"])

    #map(y->map(x->push!(trials[x],y),data["N"][y]),1:data["trial0"])
    #map(n-> map(t-> append!(data["spike_counts_by_neuron"][n],
    #            data["spike_counts"][t][n]), 1:data["ntrials"]), 1:data["N"])

    #return trials, SC

    #=

    if (haskey(data,"spike_counts_stimulus_aligned_extended_by_neuron") |
        haskey(data, "spike_counts_cpoke_aligned_extended_by_neuron"))

        #trials = Vector{Vector{Int}}()
        data["spike_counts_stimulus_aligned_extended_by_neuron"] = Vector{Vector{Vector{Int64}}}()

        #map(x->push!(trials,Vector{Int}(undef,0)),1:data["N0"])
        map(x-> push!(data["spike_counts_stimulus_aligned_extended_by_neuron"], Vector{Vector{Int}}(undef,0)), 1:data["N0"])

        #map(y->map(x->push!(trials[x],y),data["N"][y]),1:data["trial0"])
        map(n-> map(t-> append!(data["spike_counts_stimulus_aligned_extended_by_neuron"][n],
            data["spike_counts_stimulus_aligned_extended"][t][data["N"][t] .== n]), data["trial"][n]), 1:data["N0"])

        #trials = Vector{Vector{Int}}()
        data["spike_counts_cpoke_aligned_extended_by_neuron"] = Vector{Vector{Vector{Int64}}}()
        #map(x->push!(trials,Vector{Int}(undef,0)),1:data["N0"])
        map(x-> push!(data["spike_counts_cpoke_aligned_extended_by_neuron"], Vector{Vector{Int}}(undef,0)), 1:data["N0"])

        #map(y->map(x->push!(trials[x],y),data["N"][y]),1:data["trial0"])
        map(n-> map(t-> append!(data["spike_counts_cpoke_aligned_extended_by_neuron"][n],
            data["spike_counts_cpoke_aligned_extended"][t][data["N"][t] .== n]), data["trial"][n]), 1:data["N0"])

    end

    =#

    #return data

#end

function train_test_divide(data,λ,frac)

    divided_data = Dict("train" => Dict(), "test" => Dict())
    divided_λ = Dict("train" => Dict(), "test" => Dict())

    tot_trials = vcat(data["trial"]...)

    train_trials = StatsBase.sample(tot_trials, Int(ceil(frac * length(tot_trials))), replace = false)
    test_trials = setdiff(tot_trials, train_trials)

    my_keys = collect(keys(divided_data))

    for i = 1:length(my_keys)

        my_keys[i] == "train" ? trials = train_trials : trials = test_trials

        divided_data[my_keys[i]]["trial"] = [1:length(trials)]

        divided_data[my_keys[i]]["binned_leftbups"] = data["binned_leftbups"][trials]
        divided_data[my_keys[i]]["binned_rightbups"] = data["binned_rightbups"][trials]
        divided_data[my_keys[i]]["rightbups"] = data["rightbups"][trials]
        divided_data[my_keys[i]]["leftbups"] = data["leftbups"][trials]
        divided_data[my_keys[i]]["T"] = data["T"][trials]
        divided_data[my_keys[i]]["nT"] = data["nT"][trials]
        divided_data[my_keys[i]]["pokedR"] = data["pokedR"][trials]
        divided_data[my_keys[i]]["correct_dir"] = data["correct_dir"][trials]
        divided_data[my_keys[i]]["sessID"] = data["sessID"][trials]
        divided_data[my_keys[i]]["ratID"] = data["ratID"][trials]
        divided_data[my_keys[i]]["stim_start"] = data["stim_start"][trials]
        divided_data[my_keys[i]]["N"] = data["N"][trials]
        divided_data[my_keys[i]]["spike_counts"] = data["spike_counts"][trials]
        divided_data[my_keys[i]]["cellID"] = data["cellID"][trials]
        divided_data[my_keys[i]]["N0"] = data["N0"]
        divided_data[my_keys[i]]["dt"] = data["dt"]

        divided_data[my_keys[i]]["spike_counts_by_neuron"] = map(x-> x[trials], data["spike_counts_by_neuron"])

        divided_λ[my_keys[i]]["by_neuron"] = map(x-> x[trials], λ["by_neuron"])
        divided_λ[my_keys[i]]["by_trial"] = λ["by_trial"][trials]

    end

    return divided_data, divided_λ

end

function train_test_divide_multi(data,λ,frac)

    divided_data = Dict("train" => Dict(), "test" => Dict())
    divided_λ = Dict("train" => Dict(), "test" => Dict())

    my_keys = collect(keys(divided_data))

    #tot_trials = 1:data["trial0"]

    #train_trials = sort(StatsBase.sample(tot_trials, Int(ceil(frac * length(tot_trials))), replace = false))
    #test_trials = sort(setdiff(tot_trials, train_trials))

    blah = fill!(Array{Float64,2}(undef,data["N0"],2),0.)
    tot_trials = 1:data["trial0"]

    while any(blah .== 0)

        #not sure abou this...
        global train_trials = sort(StatsBase.sample(tot_trials, Int(ceil(frac * length(tot_trials))), replace = false))
        global test_trials = sort(setdiff(tot_trials, train_trials));

        for i = 1:data["N0"]
            blah[i,1] = length(intersect(data["trial"][i], train_trials))
            blah[i,2] = length(intersect(data["trial"][i], test_trials))
        end
    end

    for i = 1:length(my_keys)

        my_keys[i] == "train" ? trials = train_trials : trials = test_trials

        divided_data[my_keys[i]]["trial0"] = 0

        divided_data[my_keys[i]]["binned_leftbups"] = data["binned_leftbups"][trials]
        divided_data[my_keys[i]]["binned_rightbups"] = data["binned_rightbups"][trials]
        divided_data[my_keys[i]]["rightbups"] = data["rightbups"][trials]
        divided_data[my_keys[i]]["leftbups"] = data["leftbups"][trials]
        divided_data[my_keys[i]]["T"] = data["T"][trials]
        divided_data[my_keys[i]]["nT"] = data["nT"][trials]
        divided_data[my_keys[i]]["pokedR"] = data["pokedR"][trials]
        divided_data[my_keys[i]]["correct_dir"] = data["correct_dir"][trials]
        divided_data[my_keys[i]]["sessID"] = data["sessID"][trials]
        divided_data[my_keys[i]]["ratID"] = data["ratID"][trials]
        divided_data[my_keys[i]]["stim_start"] = data["stim_start"][trials]
        divided_data[my_keys[i]]["N"] = data["N"][trials]
        divided_data[my_keys[i]]["spike_counts"] = data["spike_counts"][trials]
        divided_data[my_keys[i]]["cellID"] = data["cellID"][trials]
        divided_data[my_keys[i]]["N0"] = data["N0"]
        divided_data[my_keys[i]]["dt"] = data["dt"]

        divided_λ[my_keys[i]]["by_trial"] = λ["by_trial"][trials]

        temp = Vector{Vector{Int}}(undef,  divided_data[my_keys[i]]["N0"])
        map(k-> temp[k] = Vector{Int}(), 1:divided_data[my_keys[i]]["N0"]);

        for k = 1:length(divided_data[my_keys[i]]["N"])
            for j = 1:length(divided_data[my_keys[i]]["N"][k])
                push!(temp[divided_data[my_keys[i]]["N"][k][j]], Int(k))
            end
        end

        divided_data[my_keys[i]]["trial"] = Vector{UnitRange{Int64}}(undef, divided_data[my_keys[i]]["N0"])

        for k = 1:divided_data[my_keys[i]]["N0"]
           divided_data[my_keys[i]]["trial"][k] = temp[k][1]:temp[k][end]
        end

        divided_data[my_keys[i]]["spike_counts_by_neuron"] = Vector{Vector{Vector{Int64}}}()
        divided_λ[my_keys[i]]["by_neuron"] = Vector{Vector{Vector{Float64}}}()

        map(x-> push!(divided_data[my_keys[i]]["spike_counts_by_neuron"],
                Vector{Vector{Int}}(undef,0)), 1:divided_data[my_keys[i]]["N0"])
        map(x-> push!(divided_λ[my_keys[i]]["by_neuron"],
                Vector{Vector{Float64}}(undef,0)), 1:divided_data[my_keys[i]]["N0"])

        for j = 1:divided_data[my_keys[i]]["N0"]

            ntrials = length(intersect(data["trial"][j],trials))
            divided_data[my_keys[i]]["trial0"] += ntrials

            map(t-> append!(divided_data[my_keys[i]]["spike_counts_by_neuron"][j],
                divided_data[my_keys[i]]["spike_counts"][t][divided_data[my_keys[i]]["N"][t] .== j]),
                divided_data[my_keys[i]]["trial"][j])

            map(t-> append!(divided_λ[my_keys[i]]["by_neuron"][j],
                divided_λ[my_keys[i]]["by_trial"][t][divided_data[my_keys[i]]["N"][t] .== j]),
                divided_data[my_keys[i]]["trial"][j])

        end

    end

    return divided_data, divided_λ

end

function package_extended_data!(data,rawdata,model_type::String,ratname,ts::Float64;dt::Float64=2e-2,organize::String="by_trial")

    ntrials = length(rawdata["T"])

    append!(data["sessid"],map(x->x[1],rawdata["sessid"]))
    append!(data["cell"],map(x->vec(collect(x)),rawdata["cell"]))
    append!(data["ratname"],map(x->ratname,rawdata["cell"]))

    maxT = ceil.(Int,(rawdata["T"])/dt)
    binnedT = ceil.(Int,(rawdata["T"] + ts)/dt);

    append!(data["nT"],binnedT)

    if any(model_type .== "spikes")

        N = size(rawdata["St"][1],2)

        if organize == "by_trial"

            append!(data["spike_counts"],map((x,y)->map(z->fit(Histogram,vec(collect(y[z])),
                    -ts:dt:x*dt,closed=:left).weights,1:N),maxT,rawdata["St"]));
            append!(data["N"],repmat([collect(data["N0"]+1:data["N0"]+N)],ntrials));

        elseif organize == "by_neuron"

            append!(data["spike_counts"],map!(z -> map!((x,y) -> fit(Histogram,vec(collect(y[z])),
                    -ts:dt:x*dt,closed=:left).weights,Vector{Vector}(ntrials),
                    maxT,rawdata["St"]),Vector{Vector}(N),1:N));
            append!(data["trial"],repmat([data["trial0"]+1:data["trial0"]+ntrials],N));

        end

        data["N0"] += N
        data["trial0"] += ntrials

    end

    return data

end

#scrub a larger dataset to only keep data relevant to a single neuron
function keep_single_neuron_data!(data,i)

    data["nT"] = data["nT"][data["trial"][i]]
    data["leftbups"] = data["leftbups"][data["trial"][i]]
    data["rightbups"] = data["rightbups"][data["trial"][i]]
    data["binned_rightbups"] = data["binned_rightbups"][data["trial"][i]]
    data["binned_leftbups"] = data["binned_leftbups"][data["trial"][i]]

    data["N"] = data["N"][data["trial"][i]]
    data["spike_counts"] = data["spike_counts"][data["trial"][i]]

    data["spike_counts"] = map((x,y)->x = x[y.==i],data["spike_counts"],data["N"])
    map!(x->x = [1],data["N"],data["N"])

    return data

end

#just hanging on to this for some later time
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

#function my_callback(os)

    #so_far = time() - start_time
    #println(" * Time so far:     ", so_far)

    #history = Array{Float64,2}(sum(fit_vec),0)
    #history_gx = Array{Float64,2}(sum(fit_vec),0)
    #for i = 1:length(os)
    #    ptemp = group_params(os[i].metadata["x"], p_const, fit_vec)
    #    ptemp = map_func!(ptemp,model_type,"tanh",N=N)
    #    ptemp_opt, = break_params(ptemp, fit_vec)
    #    history = cat(2,history,ptemp_opt)
    #    history_gx = cat(2,history_gx,os[i].metadata["g(x)"])
    #end
    #print(os[1]["x"])
    #save(ENV["HOME"]*"/spike-data_latent-accum"*"/history.jld", "os", os)
    #print(path)

#    return false

#end

=#
