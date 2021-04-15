"""
    simulate_model(model)

Simulate expected firing rates `λ` and choice given an instance of [`jointDDM`](@ref)

Arguments:

-`model`: an instance of a `jointDDM`.

Optional arguments:

-`num_samples`: How many independent samples of the latent to simulate, to average over
-`nconds`: Number of groups
-`seed`: integer for reseeding the random number generator

Returns:

-` λ`: a four-tier nested array. The outermost array is of length number of trialsets. Each entry is an `array` of length number of trials. Each entry of the subarray is more nested `array` of length number of neurons. Each entry of the sub-subarray is an array of length number of time bins. Each entry of the innermost array indicates the expected firing rate of a neuron.
-`fraction_right`: the fraction of samples resulting in a right choice. A two-tier nested vector whose outer array is of length number of trialsets and inner array is of length number of trials

"""
function simulate_model(model::jointDDM; num_samples::Int=100, seed::Int=1)

    @unpack θ, joint_data = model
    @unpack θz, θy, θh, bias, lapse = θ
    @unpack B = θz
    shifted = map(x->x.shifted, joint_data)
    neural_data= map(x->x.neural_data, joint_data)

    a₀ = map(x->history_influence_on_initial_point(θh, B, x), shifted)
    seeds = sample(Random.seed!(seed), 1:num_samples, num_samples; replace=false)
    λ, choseright = map(x-> rand.(Ref(θz), θy, bias, lapse, neural_data, a₀, Ref(x)), seeds)

    return mean(λ), mean(choseright)
end

"""
    rand(θz, θy, neural_data, a₀, seed)

Sample all trials in one trialset.

Arguments:
-`θz`: an instance of θz, which includes the parameters σ2_i, σ2_s, σ2_a, λ, B, ϕ, and τ_ϕ
-`θy`: an instance of θy, which specifies the mapping from the latent to the firing rates
-`bias`: A float
-`lapse`: A float in the interval [0,1]
-`neural_data`: a vector of [`neuraldata`](@ref)
-`a₀`: A vector of Floats specifying the initial point
-`seed`: an integer

Returns:
-`λ`: a three-tier nested array. The outermost array is of length number of trials. Each entry of the subarray is a more nested array of length number of neurons. Each entry of the sub-subarray is an even more nested array of length number of time bins. Each entry of the innermost array indicates the expected firing rate of a neuron.
-`choseright`: an array of length number of trials and whose each entry is a Bool
"""
function rand(θz::θz, θy, bias::T2, lapse::T2, neural_data::Vector{T1}, a₀::Vector{T2}, seed::Int) where {T1 <: neuraldata, T2<:AbstractFloat}
    ntrials = length(data)
    seeds = sample(Random.seed!(seed), 1:ntrials, ntrials; replace=false)
    pmap((data,a₀,seeds) -> rand(θz, θy, bias, lapse, data.input_data, a₀, seeds), neural_data, a₀, seeds)
end

"""
    rand(θz, θy, input_data, a₀, seed)

Sample a single trial

Arguments:
-`θz`: an instance of θz, which includes the paramters σ2_i, σ2_s, σ2_a, λ, B, ϕ, and τ_ϕ
-`θy`: an instance of θy, which specifies the mapping from the latent to the firing rates
-`bias`: A float
-`lapse`: A float in the interval [0,1]
-`input_data`: instance of [`neuralinputs`](@ref)
-`a₀`: A Float specifying the initial point
-`seed`: an integer

Returns:
-`λ`: The expected firing rate of each neuron. It is a two-tier nested array whose outer array is of length number of neuron and inner array of length number of time bins.
-`choseright`: A Bool indicating whether a right choice was made
"""
function rand(θz::θz, θy, bias::T2, lapse::T2, input_data::neuralinputs, a₀, seed::Int=1)

    @unpack λ0, dt = input_data

    Random.seed!(seed)
    a = rand(θz, input_data, a₀)
    λ = map((θy,λ0)-> θy(a, λ0), θy, λ0)
    #spikes = map(λ-> rand.(Poisson.(λ*dt)), λ)
    choseright = rand() > lapse ? a[end] > bias : rand() > 0.5

    return λ, choseright
end

"""
    rand(θz, inputs)

Generate the trajectory of the latent variable during a trial

Arguments:
-`θz`: an instance of θz, which includes the paramters σ2_i, σ2_s, σ2_a, λ, B, ϕ, and τ_ϕ
-`inputs`: an instance of `neuralinputs`
-`a₀`: initial value of the latent

Returns:
- `A`: an `array` of the latent path.

"""
function rand(θz::θz{T}, inputs::neuralinputs, a₀::T) where T <: Real

    @unpack σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ = θz
    @unpack clicks, binned_clicks, centered, dt, delay, pad = inputs
    @unpack nT, nL, nR = binned_clicks
    @unpack L, R = clicks

    La, Ra = adapt_clicks(ϕ, τ_ϕ, L, R)
    time_bin = (-(pad-1):nT+pad) .- delay
    A = Vector{T}(undef, length(time_bin))
    if σ2_i > 0.
        a = a₀ + sqrt(σ2_i)*randn()
    else
        a = a₀
    end
    for t = 1:length(time_bin)
         if time_bin[t] < 1
            if σ2_i > 0.
                a = a₀ + sqrt(σ2_i)*randn()
            else
                a = a₀
            end
        else
            a = sample_one_step!(a, time_bin[t], σ2_a, σ2_s, λ, nL, nR, La, Ra, dt)
        end
        abs(a) > B ? (a = B * sign(a); A[t:end] .= a; break) : A[t] = a
    end
    return A
end
