"""
    reload_joint_model(file)

`reload_joint_model` will bring back the parameters from your fit, some details about the optimization (such as the `fit` and bounds vectors) and some details about how you filtered the data. All of the data is not saved in the format that it is loaded by `load_neural_data` because it's too cumbersome to seralize it, so you have to load it again, as above, to re-build `neuralDDM` but you can use some of the stuff that `reload_neural_data` returns to reload the data in the same way (such as `pad` and `dt`)

Returns:

- `θneural`
- `neural_options`
- `n`
- `cross`
- `dt`
- `delay`
- `pad`

See also: [`save_neural_model`](@ref)

"""
function reload_joint_model(file)

    xf = read(matopen(file), "ML_params")
    f = string.(read(matopen(file), "f"))
    ncells = collect(read(matopen(file), "ncells"))
    nparams = read(matopen(file), "nparams")
        
    borg = vcat(0,cumsum(ncells, dims=1))
    nparams = [nparams[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]
    f = [f[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]   
    
    lb = read(matopen(file), "lb")
    ub = read(matopen(file), "ub")
    fit = read(matopen(file), "fit")
    
    n = read(matopen(file), "n")
    cross = read(matopen(file), "cross")
    dt = read(matopen(file), "dt")
    delay = read(matopen(file), "delay")
    pad = read(matopen(file), "pad")       
    
    θneural_choice(xf, f), neural_choice_options(lb=lb, ub=ub, fit=fit), n, cross, dt, delay, pad 
    
end


"""
    load_neural_data(path, files)

Load neural data .MAT files and return a Dict.
"""
function load_neural_choice(file::String, break_sim_data::Bool, centered::Bool=true;
        dt::Float64=1e-2, delay::Int=0, pad::Int=20, filtSD::Int=5,
        extra_pad::Int=10, cut::Int=10, pcut::Float64=0.01, include_choices::Bool=true)

    data = read(matopen(file), "rawdata")

    T = vec(data["T"])
    L = vec(map(x-> vec(collect(x)), data[collect(keys(data))[occursin.("left", collect(keys(data)))][1]]))
    R = vec(map(x-> vec(collect(x)), data[collect(keys(data))[occursin.("right", collect(keys(data)))][1]]))
    choices = vec(convert(BitArray, data["pokedR"]))

    click_times = clicks.(L, R, T)
    binned_clicks = bin_clicks(click_times, centered=centered, dt=dt)
    nT = map(x-> x.nT, binned_clicks)

    ncells = size(data["spike_times"][1], 2)
    
    spikes = vec(map(x-> vec(vec.(collect.(x))), data["spike_times"]))

    output = map((spikes, nT)-> bin_spikes(spikes, dt, nT; pad=0), spikes, nT)

    spikes = getindex.(output, 1)     
    FR = map(i-> map((x,T)-> sum(x[i])/T, spikes, T), 1:ncells)
    choice = vec(convert(BitArray, data["pokedR"]))
    pval = map(x-> pvalue(EqualVarianceTTest(x[choice], x[.!choice])), FR)      
    ptest = pval .< pcut
    
    if any(ptest)
        
        ncells = sum(ptest)

        if break_sim_data

            spike_data = Vector{Vector{neuraldata}}(undef, ncells)
            μ_rnt = Vector(undef, ncells)
            μ_t = Vector(undef, ncells)

            for n = 1:ncells
                             
                spikes = vec(map(x-> [vec(collect(x[findall(ptest)][n]))], data["spike_times"]))

                output = map((spikes, nT)-> bin_spikes(spikes, dt, nT; pad=pad), spikes, nT)

                spikes = getindex.(output, 1)
                padded = getindex.(output, 2)  

                μ_rnt[n] = filtered_rate.(padded, dt; filtSD=filtSD, cut=cut)

                μ_t[n] = map(n-> [max(0., mean([μ_rnt[n][i][1][t]
                    for i in findall(nT .+ 2*pad .>= t)]))
                    for t in 1:(maximum(nT) .+ 2*pad)], n:n)

                λ0 = map(nT-> bin_λ0(μ_t[n], nT+2*pad), nT)
                #λ0 = map(nT-> map(μ_t-> zeros(nT), μ_t), nT)

                input_data = neuralinputs(click_times, binned_clicks, λ0, dt, centered, delay, pad)
                spike_data[n] = neuraldata(input_data, spikes, 1)
                
                nRBFs=6
                model, = optimize([spike_data[n]], μ_RBF_options(ncells=[1], nRBFs=nRBFs); show_trace=false)
                maxnT = maximum(nT)
                x = 1:maxnT+2*pad   
                rbf = UniformRBFE(x, nRBFs, normalize=true)  
                μ_t[n] = [rbf(x) * model.θ.θμ[1][1]]
                    
                #model, = optimize([spike_data[n]], μ_poly_options(ncells=[1]); show_trace=false)
                #μ_t[n] = [model.θ.θμ[1][1](1:length(μ_t[n][1]))]
                    
                λ0 = map(nT-> bin_λ0(μ_t[n], nT+2*pad), nT)      
                input_data = neuralinputs(click_times, binned_clicks, λ0, dt, centered, delay, pad)
                
                if include_choices
                    spike_data[n] = neural_choice_data(input_data, choices, spikes, 1)
                else
                    spike_data[n] = neuraldata(input_data, spikes, 1)
                end

            end

        else

            spikes = vec(map(x-> vec(vec.(collect.(x))), data["spike_times"]))

            output = map((spikes, nT)-> bin_spikes(spikes, dt, nT; pad=pad), spikes, nT)

            spikes = getindex.(output, 1)
            padded = getindex.(output, 2)      

            spikes = map(spikes-> spikes[ptest], spikes)
            padded = map(padded-> padded[ptest], padded)

            μ_rnt = filtered_rate.(padded, dt; filtSD=filtSD, cut=cut)

            μ_t = map(n-> [max(0., mean([μ_rnt[i][n][t]
                for i in findall(nT .+ 2*pad .>= t)]))
                for t in 1:(maximum(nT) .+ 2*pad)], 1:ncells)

            #μ_t = map(n-> [max(0., mean([spikes[i][n][t]/dt
            #    for i in findall(nT .>= t)]))
            #    for t in 1:(maximum(nT))], 1:ncells)

            λ0 = map(nT-> bin_λ0(μ_t, nT+2*pad), nT)
            #λ0 = map(nT-> map(μ_t-> zeros(nT), μ_t), nT)

            input_data = neuralinputs(click_times, binned_clicks, λ0, dt, centered, delay, pad)
            spike_data = neuraldata(input_data, spikes, ncells)
            
            nRBFs=6
            model, = optimize([spike_data], μ_RBF_options(ncells=[ncells], nRBFs=nRBFs); show_trace=false)
            maxnT = maximum(nT)
            x = 1:maxnT+2*pad   
            rbf = UniformRBFE(x, nRBFs, normalize=true) 
            μ_t = map(n-> rbf(x) * model.θ.θμ[1][n], 1:ncells)
            
            #model, = optimize([spike_data], μ_poly_options(ncells=[ncells]); show_trace=false)
            #μ_t = map(n-> model.θ.θμ[1][n](1:length(μ_t[n])), 1:ncells)
            
            λ0 = map(nT-> bin_λ0(μ_t, nT+2*pad), nT)
            input_data = neuralinputs(click_times, binned_clicks, λ0, dt, centered, delay, pad)
            if include_choices
                spike_data = neural_choice_data(input_data, choices, spikes, ncells)
            else
                spike_data = neuraldata(input_data, spikes, ncells)
            end


        end
        
        return spike_data, μ_rnt, μ_t
            
    else

        return nothing

    end
 
end