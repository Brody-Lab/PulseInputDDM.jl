"""
    load_joint_data(file::String, ...

Load data for fitting the `jointDDM` model. See (['load_neural_data']@ref)) for information about the optional arguments

Arguments:

- `file`: string, the full path of the MATLAB ".mat" file containing the behavioral and neural data corresponding to single `trial-set`. A trial-set is used indicate a group of (not necessarily ordered) trials on which the same set of neurons were recorded. The trials could come from multiple discontinous daily sessions if neurons were tracked across days.

Optional arguments:
- `nback`: number of trials in the past to consider

Returns:

- `joint_data`: an instance of [`jointdata`](@ref)
- `μ_rnt`: an array of length number of trials. Each entry of the sub-array is also an `array`, of length number of cells. Each entry of that array is the filtered single-trial firing rate of each neuron
- `μ_t`: an `array` of length number of cells. Each entry is the trial-averaged firing rate (across all trials).

"""
function load_joint_data(file::String; break_sim_data::Bool=false,
        dt::Float64=1e-2, delay::Int=6, pad::Int=20, filtSD::Int=2,
        extra_pad::Int=10, cut::Int=10, pcut::Float64=0.0001,
        do_RBF::Bool=false, nRBFs::Int=6, centered::Bool=true, nback::Int=10)

    output = load_neural_data(file; break_sim_data=break_sim_data, dt = dt, delay = delay, pad = pad, filtSD = filtSD, extra_pad = extra_pad)
    if ~isnothing(output)
        spike_data = output[1]
        μ_rnt = output[2]
        μ_t = output[3]
        sequence = load_trial_sequence(file)
        shifted = get_trialshifted(sequence, nback)
        @assert check_trialsequence_matches_neuraldata(sequence, spike_data)
        joint_data = jointdata(spike_data, sequence, shifted)
        return joint_data, μ_rnt, μ_t
    end
end

"""
    load_joint_data(file::Vector{String}, ...

Calls `load_joint_data` for each entry in `file` and then creates three array outputs—`joint_data`, `μ_rnt`, `μ_t`—where each entry of an array is the relevant data for a single file. In each file, all the neurons are expected to have been recorded simultaneously.

Returns:

- `joint_data`: an `array` of length number of trial-sets. Each entry is for a trial-set, and is another `array`. Each entry of the sub-array is the relevant data for a trial.
- `μ_rnt`: an `array` of length number of trial-sets. Each entry is another `array` of length number of trials. Each entry of the sub-array is also an `array`, of length number of cells. Each entry of that array is the filtered single-trial firing rate of each neuron
- `μ_t`: an `array` of length number of trial-sets. Each entry is an `array` of length number of cells. Each entry is the trial-averaged firing rate (across all trials).
"""
function load_joint_data(file::Vector{String}; break_sim_data::Bool=false,
        centered::Bool=true, dt::Float64=1e-2, delay::Int=6, pad::Int=20, filtSD::Int=2,
        extra_pad::Int=10, cut::Int=10, pcut::Float64=0.0001,
        do_RBF::Bool=false, nRBFs::Int=6, nback::Int=10)

    output = load_joint_data.(file; break_sim_data=break_sim_data,
        centered=centered,
        dt=dt, delay=delay, pad=pad, filtSD=filtSD,
        extra_pad=extra_pad, cut=cut, pcut=pcut,
        do_RBF=do_RBF, nRBFs=nRBFs, nback=nback)

    output = filter(x -> x != nothing, output)

    joint_data = getindex.(output, 1)
    μ_rnt = getindex.(output, 2)
    μ_t = getindex.(output, 3)

    joint_data, μ_rnt, μ_t
end

"""
    load_trial_sequence

Load the trial sequence from the full path of a MATLAB ".mat" file

"""
function load_trial_sequence(file::String)
    sequence = read(matopen(file), "trialsequence")

    choice = sequence["choice"][:]
    ignore = sequence["ignore"][:]
    index = convert.(Int, sequence["index"][:])
    reward = sequence["reward"][:]
    sessionstart = sequence["sessionstart"][:]

    @assert length(choice) == length(reward) == length(sessionstart)
    @assert minimum(index) >= 1
    @assert maximum(index) <= length(choice)
    @assert ~any(ignore[index]) "trials whose choice and spikes are being fitted cannot be ignored"

    trialsequence(choice = choice,
                  ignore = ignore,
                  index = index,
                  reward = reward,
                  sessionstart = sessionstart)
end

"""
    get_trialshifted

Return `trialshifted` information from `trialsequence`

Arguments:

- `sequence`: a `trialsequence` object
- `nback` Maximum number of trials to get values from

Returns:

- `trialshifted` ([`trialshifted`](@ref))
"""
function get_trialshifted(sequence::trialsequence, nback::Int)
    @assert nback > 0
    @unpack sessionstart, choice, ignore, reward, index = sequence
    choice = 2*choice .- 1;
    reward = 2*reward .- 1;
    choice[ignore] .= 0;
    reward[ignore] .= 0;

    n = length(choice)
    pastchoices = Array{Int}(undef,n,nback)
    pastrewards = Array{Int}(undef,n,nback)
    startindices = findall(sessionstart)
    for i in eachindex(startindices)
        i1st = startindices[i]
        if i == length(startindices)
            iend = length(choice)
        else
            iend = startindices[i+1]-1;
        end
        pastchoices[i1st:iend,:] = get_past_values(choice[i1st:iend], nback)
        pastrewards[i1st:iend,:] = get_past_values(reward[i1st:iend], nback)
    end
    pastchoices = pastchoices[index, :]
    pastrewards = pastrewards[index, :]
    shifts = collect(-1:-1:-nback)
    shifts = reshape(shifts, (1,length(shifts)))
    trialshifted(choice=pastchoices, reward=pastrewards, shift=shifts)
end

"""
    get_past_values

Returns back-shifted values of each element of a vector of numbers

Argument:

- `x`: a vector of numbers
- 'n': number of back-shift values

Optional argument:

-`miss`: value used for elements representing a lack of backshifted information
"""
function get_past_values(x::Vector{T}, n::Int; miss = 0) where T <: Real
    miss = convert(T, miss)
    x = convert.(Float64, x) # The `Hankel` method in ToeplitzMatrices.jl takes only with Float64
    if length(x) <= n
        x = vcat(x, repeat([miss], n-length(x)+1))
    end
    first_column = vcat(repeat([miss], n), x[1:end-n])
    last_row = x[end-n:end-1]
    M = Hankel(first_column, last_row)
    M = reverse(M, dims = 2)
    M = M[1:length(x),:]
    if T != Float64
        M = convert(Array{T}, M)
    else
        return M
    end
end

"""
    check_trialsequence_matches_neuraldata

Returns true if the length of a vector of `neuraldata` matches the length of the field `index` in an instance `trialsequence`
"""
function check_trialsequence_matches_neuraldata(sequence::trialsequence, neural_data::Vector{neuraldata})
    @unpack index = sequence
    length(neural_data) == length(index)
end