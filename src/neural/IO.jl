"""
    save_neural_model(file, model, options)

Given a `file`, `model` and `options` produced by `optimize`, save everything to a `.MAT` file in such a way that `reload_neural_data` can bring these things back into a Julia workspace, or they can be loaded in MATLAB.

See also: [`reload_neural_model`](@ref)

"""
function save_neural_model(file, model::Union{neuralDDM, neural_choiceDDM}, data)

    @unpack θ, n, cross, lb, ub, fit = model
    @unpack f = θ
    @unpack dt, delay, pad = data[1][1].input_data
    
    nparams, ncells = nθparams(f)
    
    dict = Dict("ML_params"=> collect(PulseInputDDM.flatten(θ)),
        "lb"=> lb, "ub"=> ub, "fit"=> fit, "n"=> n, "cross"=> cross,
        "dt"=> dt, "delay"=> delay, "pad"=> pad, "f"=> vcat(vcat(f...)...),
        "nparams" => nparams, "ncells" => ncells)

    matwrite(file, dict)

end


"""
    load_neural_model(file)

`reload_neural_data` will bring back the parameters from your fit, some details about the optimization (such as the `fit` and bounds vectors) and some details about how you filtered the data. All of the data is not saved in the format that it is loaded by `load_neural_data` because it's too cumbersome to seralize it, so you have to load it again, as above, to re-build `neuralDDM` but you can use some of the stuff that `reload_neural_data` returns to reload the data in the same way (such as `pad` and `dt`)

Returns:

- `θneural`
- `neural_options`
- n
- cross
- dt
- delay
- pad

See also: [`save_neural_model`](@ref)

"""
function load_neural_model(file)

    xf = read(matopen(file), "ML_params")
    f = string.(read(matopen(file), "f"))
    ncells = collect(read(matopen(file), "ncells"))
    nparams = read(matopen(file), "nparams")
        
    borg = vcat(0,cumsum(ncells, dims=1))
    nparams = [nparams[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]
    f = [f[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]   
    
    lb = read(matopen(file), "lb")
    ub = read(matopen(file), "ub")
    fitbool = read(matopen(file), "fit")
    
    n = read(matopen(file), "n")
    cross = read(matopen(file), "cross")
    dt = read(matopen(file), "dt")
    delay = read(matopen(file), "delay")
    pad = read(matopen(file), "pad")       
    
    neuralDDM(θ=θneural(xf, f),fit=fitbool,lb=lb,ub=ub, n=n, cross=cross)
    
end


"""
    load_neural_data(file::Vector{String}; centered, dt, delay, pad, filtSD, extra_pad, cut, pcut)

Calls `load_neural_data` for each entry in `file` and then creates three array outputs—`spike_data`, `μ_rnt`, `μ_t`—where each entry of an array is the relevant data for a single session. 

Returns:

- `data`: an `array` of length number of session. Each entry is for a session, and is another `array`. Each entry of the sub-array is the relevant data for a trial.
- `μ_rnt`: an `array` of length number of sessions. Each entry is another `array` of length number of trials. Each entry of the sub-array is also an `array`, of length number of cells. Each entry of that array is the filtered single-trial firing rate of each neuron
- `μ_t`: an `array` of length number of sessions. Each entry is an `array` of length number of cells. Each entry is the trial-averaged firing rate (across all trials).

"""
function load_neural_data(file::Vector{String}; break_sim_data::Bool=false, 
        centered::Bool=true, dt::Float64=1e-2, delay::Int=0, pad::Int=0, filtSD::Int=2,
        extra_pad::Int=10, cut::Int=10, pcut::Float64=0.01, 
        do_RBF::Bool=true, nRBFs::Int=6)
    
    output = load_neural_data.(file; break_sim_data=break_sim_data,
        centered=centered,
        dt=dt, delay=delay, pad=pad, filtSD=filtSD,
        extra_pad=extra_pad, cut=cut, pcut=pcut, 
        do_RBF=do_RBF, nRBFs=nRBFs)
    
    output = filter(x -> x != nothing, output)
    
    spike_data = getindex.(output, 1)
    μ_rnt = getindex.(output, 2)
    μ_t = getindex.(output, 3)  
    cpoke_out = getindex.(output, 4)  
    
    spike_data, μ_rnt, μ_t, cpoke_out
    
end

"""
    load_neural_data(file::String; centered, dt, delay, pad, filtSD, extra_pad, cut, pcut)

Load neural data `.MAT` file and return an three `arrays`. The first `array` is the `data` formatted correctly for fitting the model. Each element of `data` is a module-defined class called `neuraldata`.

The package expects your data to live in a single `.MAT` file which should contain a struct called `rawdata`. Each element of `rawdata` should have data for one behavioral trial and `rawdata` should contain the following fields with the specified structure:

- `rawdata.leftbups`: row-vector containing the relative timing, in seconds, of left clicks on an individual trial. 0 seconds is the start of the click stimulus
- `rawdata.rightbups`: row-vector containing the relative timing in seconds (origin at 0 sec) of right clicks on an individual trial. 0 seconds is the start of the click stimulus.
- `rawdata.T`: the duration of the trial, in seconds. The beginning of a trial is defined as the start of the click stimulus. The end of a trial is defined based on the behavioral event “cpoke_end”. This was the Hanks convention.
- `rawdata.pokedR`: Bool representing the animal choice (1 = right).
- `rawdata.spike_times`: cell array containing the spike times of each neuron on an individual trial. The cell array will be length of the number of neurons recorded on that trial. Each entry of the cell array is a column vector containing the relative timing of spikes, in seconds. Zero seconds is the start of the click stimulus. Spikes before and after the click inputs should also be included.

Arguments:

- `file`: path to the file you want to load.

Optional arguments;

- `break_sim_data`: this will break up simulatenously recorded neurons, as if they were recorded independently. Not often used by most users.
- `centered`: Defaults to true. For the neural model, this aligns the center of the binned spikes, to the beginning of the binned clicks. This was done to fix a numerical problem. Most users will never need to adjust this. 
- `dt`: Binning of the spikes, in seconds.
- `delay`: How much to offset the spikes, relative to the accumlator, in units of `dt`.
- `pad`: How much extra time should spikes be considered before and after the begining of the clicks. Useful especially if delay is large.
- `filtSD`: standard deviation of a Gaussin (in units of `dt`) to filter the spikes with to generate single trial firing rates (`μ_rnt`), and mean firing rate across all trials (`μ_t`).
- `extra_pad`: Extra padding (in addition to `pad`) to add, for filtering purposes. In units of `dt`.
- `cut`: How much extra to cut off at the beginning and end of filtered things (should be equal to `extra_pad` in most cases).
- `pcut`: p-value for selecting cells.

Returns:

- `data`: an `array` of length number of trials. Each element is a module-defined class called `neuraldata`.
- `μ_rnt`: an `array` of length number of trials. Each entry of the sub-array is also an `array`, of length number of cells. Each entry of that array is the filtered single-trial firing rate of each neuron.
- `μ_t`: an `array` of length number of cells. Each entry is the trial-averaged firing rate (across all trials).


"""
function load_neural_data(file::String; break_sim_data::Bool=false, 
        dt::Float64=1e-2, delay::Int=0, pad::Int=0, filtSD::Int=2,
        extra_pad::Int=10, cut::Int=10, pcut::Float64=0.01, 
        do_RBF::Bool=true, nRBFs::Int=6, centered::Bool=true)

    data = read(matopen(file), "rawdata")
    
    if !haskey(data, "spike_times")
        data["spike_times"] = data["St"]
    end
    
    if haskey(data, "cpoke_out")
        cpoke_out = data["cpoke_out"]
        cpoke_end = data["cpoke_end"]
    else
        cpoke_out = data["cpoke_end"]
        cpoke_end = data["cpoke_end"]
    end

    T = vec(data["T"])
    L = vec(map(x-> vec(collect(x)), data[collect(keys(data))[occursin.("left", collect(keys(data)))][1]]))
    R = vec(map(x-> vec(collect(x)), data[collect(keys(data))[occursin.("right", collect(keys(data)))][1]]))

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
                spike_data[n] = neuraldata(input_data, spikes, 1, choice)
                
                if do_RBF
                    θ = optimize([spike_data[n]], [1]; nRBFs=nRBFs, show_trace=false)
                    maxnT = maximum(nT)
                    x = 1:maxnT+2*pad   
                    rbf = UniformRBFE(x, nRBFs, normalize=true)  
                    μ_t[n] = [rbf(x) * θ.θμ[1][1]]
                end
                    
                #model, = optimize([spike_data[n]], μ_poly_options(ncells=[1]); show_trace=false)
                #μ_t[n] = [model.θ.θμ[1][1](1:length(μ_t[n][1]))]
                    
                λ0 = map(nT-> bin_λ0(μ_t[n], nT+2*pad), nT)      
                input_data = neuralinputs(click_times, binned_clicks, λ0, dt, centered, delay, pad)
                spike_data[n] = neuraldata(input_data, spikes, 1, choice)

            end

        else

            #=
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
            =#
            
            μ_t = Vector(undef, ncells)

            for n = 1:ncells

                spikes = vec(map(x-> [vec(collect(x[findall(ptest)][n]))], data["spike_times"]))

                output = map((spikes, nT)-> PulseInputDDM.bin_spikes(spikes, dt, nT; pad=pad), spikes, nT)

                spikes = getindex.(output, 1)
                padded = getindex.(output, 2)  

                μ_rnt = PulseInputDDM.filtered_rate.(padded, dt; filtSD=filtSD, cut=cut)

                μ_t[n] = map(n-> [max(0., mean([μ_rnt[i][1][t]
                    for i in findall(nT .+ 2*pad .>= t)]))
                    for t in 1:(maximum(nT) .+ 2*pad)], n:n)

                λ0 = map(nT-> PulseInputDDM.bin_λ0(μ_t[n], nT+2*pad), nT)

                input_data = PulseInputDDM.neuralinputs(click_times, binned_clicks, λ0, dt, centered, delay, pad)
                spike_data = PulseInputDDM.neuraldata(input_data, spikes, 1, choice)

                if do_RBF
                    θ = optimize([spike_data], [1]; nRBFs=nRBFs, show_trace=false)
                    maxnT = maximum(nT)
                    x = 1:maxnT+2*pad   
                    rbf = UniformRBFE(x, nRBFs, normalize=true)  
                    μ_t[n] = [rbf(x) * θ.θμ[1][1]]
                end

            end
            
            μ_t = map(x-> x[1], μ_t);
            
            spikes = vec(map(x-> vec(vec.(collect.(x))), data["spike_times"]))
            output = map((spikes, nT)-> bin_spikes(spikes, dt, nT; pad=pad), spikes, nT)

            spikes = getindex.(output, 1)
            padded = getindex.(output, 2)      

            spikes = map(spikes-> spikes[ptest], spikes)
            padded = map(padded-> padded[ptest], padded)

            μ_rnt = filtered_rate.(padded, dt; filtSD=filtSD, cut=cut)
            
            λ0 = map(nT-> bin_λ0(μ_t, nT+2*pad), nT)
            input_data = neuralinputs(click_times, binned_clicks, λ0, dt, centered, delay, pad)
            spike_data = neuraldata(input_data, spikes, ncells, choice)


        end
        
        return spike_data, μ_rnt, μ_t, cpoke_out - cpoke_end 
            
    else

        return nothing

    end
 
end