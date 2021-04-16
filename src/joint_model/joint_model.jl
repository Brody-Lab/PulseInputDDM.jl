"""
    jointDDM(data; ftype,remap,modeltype,n,cross)

Create an instance of the ['jointDDM'](@ref) and return the options for optimizing this model

Arguments:

-`data`: a Vector of instances of [`jointdata`](@ref). Each element correspond to a single trial-set. If the vector has multiple elements, then the same parameters controlling the latent variable are fit to all the data across the trialsets. Different parameters controlling the mapping from the latent variable to the firing rates are fit to each neuron in each trial.

Optional arguments:

-`f`: function mapping the latent variable to firing rate. It can be either "Softplus" or "Sigmoid"
-`remap`: if true, parameters are considered in variance instead of std space
-`modeltype`: a string specifying the model type. Current options include:
    "nohistory": no influence by trial history
    "history1back": the influence of only the previous trial is included
    "history": the influence of the 30 previous trials is included, and the influence further in the past decays exponentially
-`n`: Number of bins in which the latent space, a, is discretized
-`cross`: whether to adapt clicks across left and right streams, as opposed to within each stream
Returns:

-`model`: an instance [`jointDDM`](@ref)
-`options`: an instance [`joint_options`](@ref)
"""
function jointDDM(data::Vector{T1}; ftype::String="Softplus", remap::Bool=false, modeltype = :history1back, n::T2 = 53, cross::Bool = false) where {T1<:jointdata, T2<:Integer}
    @assert ftype == "Softplus" || ftype == "Sigmoid"
    θ = θjoint(data; ftype=ftype,remap=remap, modeltype=modeltype)
    model = jointDDM(θ=θ, joint_data=data, n=n, cross=cross)
end

"""
    joint_options!(options, f)

Modify an instance of [`joint_options`](@ref) to include the parameters for mapping the latent variable to firing rates

Arguments:
-`options` an instance of [`joint_options`](@ref)
-`f` A vector of vector of the String "Sigmoid" or "Softplus", specifying the mapping from latent variable to firing rate for each neuron in each trialset

Optional arguments:
-`remap`: if true, parameters are considered in variance instead of std space
-`modeltype`: a string specifying the model type. Current options include:
    "nohistory": no influence by trial history
    "history1back": the influence of only the previous trial is included
    "history": the influence of the 30 previous trials is included, and the influence further in the past decays exponentially
"""
function joint_options!(options::joint_options, f::Vector{Vector{String}})

    θlatent_fit = is_θlatent_fit_in_jointDDM(modeltype=options.modeltype)
    θlatent_lb, θlatent_ub = lookup_jointDDM_θlatent_bounds(remap=options.remap)
    θlatent_names = get_jointDDM_θlatent_names()
    n_neural_params, ncells = nθparams(f)

    fit = vcat(falses(length(θlatent_names)), trues(sum(n_neural_params)))
    lb = repeat([NaN], length(θlatent_names))
    ub = repeat([NaN], length(θlatent_names))
    for i in eachindex(θlatent_names)
        fit[i] = θlatent_fit[θlatent_names[i]]
        lb[i] = θlatent_lb[θlatent_names[i]]
        ub[i] = θlatent_ub[θlatent_names[i]]
    end

    lb_neural = Array{Vector}(undef,sum(ncells))
    ub_neural = Array{Vector}(undef,sum(ncells))
    for i in 1:sum(ncells)
        if vcat(f...)[i] == "Softplus"
            lb_neural[i] = [-10.]
            ub_neural[i] = [10.]
        elseif vcat(f...)[i] == "Sigmoid"
            lb_neural[i] = [-100.,0.,-10.,-10.]
            ub_neural[i] = [100.,100.,10.,10.]
        end
    end
    options.lb = vcat(lb, lb_neural...)
    options.ub = vcat(ub, ub_neural...)
    options.fit = fit
    return options
end
"""
    θjoint(data;ftype,remap,modeltype)

Return an instance of [`θjoint`] to be used as the initial values for model fitting. The parameters specifying the drift-diffusion process are randomly selected, and the parameters for the mapping from latent variable to firing rates result from fitting a noiseless model to the neural data

Arguments:

-is_fit: a vector of Bool specifying whether each parameter controlling the drift-diffusion is fit. The neural parameters are not included in this vector
-`f`: a vector of vector of the string either "Softplus" or "Sigmoid" specifying the type of the mapping from the latent variable to the firing rate of each neuron in each trialset

Optional arguments:
-`fit_noiseless_model`: Bool specifying whether to fit a noiseless model to find of the initial values of the  parameters mapping the latent variable to firing rates (θy)
"""

function θjoint(data::Vector{T}; ftype::String="Softplus", remap::Bool=false, modeltype::Symbol = :history1back, fit_noiseless_model::Bool=true) where {T <: jointdata}

    f = specify_a_to_firing_rate_function_type(data; ftype=ftype)
    fit = is_θlatent_fit_in_jointDDM(modeltype=modeltype)
    x0 = lookup_jointDDM_default_θlatent(remap=remap)
    lb, ub = lookup_jointDDM_θlatent_bounds(remap=remap)

    for key in collect(keys(x0))
        if fit[key]
            x0[key] = lb[key] + (ub[key] - lb[key]) * rand()
        end
    end

    initialθz = θz( σ2_i = x0[:σ2_i],
                    σ2_a = x0[:σ2_a],
                    σ2_s = x0[:σ2_s],
                    ϕ = x0[:ϕ],
                    τ_ϕ = x0[:τ_ϕ],
                    λ = x0[:λ],
                    B = x0[:B])
    initialθh = θh(α = x0[:α],
                    k = x0[:k])
    neural_data = map(x->x.neural_data, data)
    if fit_noiseless_model
        initialθy = θy0(neural_data,f)
    else
        noiselessθy = θy.(neural_data, f)
        noiselessx0 = vcat([0., 15., 0. - eps(), 0., 0., 1.0 - eps(), 0.008], vcat(vcat(noiselessθy...)...))
        noiselessθ = θneural_noiseless(noiselessx0, f)
        noiselessmodel = noiseless_neuralDDM(noiselessθ, neural_data)
        initialθy = noiselessmodel.θ.θy
    end
    θjoint( θz = initialθz,
            θh = initialθh,
            lapse = x0[:lapse],
            bias = x0[:bias],
            θy = initialθy,
            f = f)
end

"""
    lookup_jointDDM_default_θlatent(;remap)

Return the default values for the parameters controlling the latent variable for the joint model

Optional arguments:
-`remap`: whether to parametrize the noise parameters (σ2_i, σ2_a, σ2_s) as variance rather than standard deviation

Returns:
-`x0`:A Dictionary of the default initial values
"""
function lookup_jointDDM_default_θlatent(;remap::Bool=false)
    x0 = Dict(  :σ2_i => eps(),
                :σ2_a => eps(),
                :σ2_s => eps(),
                :B => 40.,
                :λ => 0.,
                :ϕ => 1.,
                :τ_ϕ => 0.1,
                :bias => 0.,
                :lapse => eps(),
                :α => 0.,
                :k => 10.)
    if remap
        map((x,y)->x0[:x]=y, [:σ2_a, :σ2_i, :σ2_s], repeat([1.],3))
    end
    return x0
end

"""
    lookup_jointDDM_θlatent_bounds(;remap)

Return the lower and upper limits of the parameters controlling the latent variable for the joint model

Optional arguments:
-`remap`: whether to parametrize the noise parameters (σ2_i, σ2_a, σ2_s) as variance rather than standard deviation

Returns:
-`lb`: a Dictionary of the lower bounds
-`ub`: a Dictionary of the upper limits
"""
function lookup_jointDDM_θlatent_bounds(;remap::Bool=false)
    lb = Dict( :σ2_i => 0.,
                    :σ2_a => 0.,
                    :σ2_s => 0.,
                    :B => 0.5,
                    :λ => -10.,
                    :ϕ => 0.01,
                    :τ_ϕ => 0.005,
                    :bias => -5.,
                    :lapse => 0.,
                    :α => -5.,
                    :k => 0.)
    ub = Dict( :σ2_i => 40.,
                   :σ2_a => 100.,
                   :σ2_s => 20.,
                   :B => 60.,
                   :λ => 10.,
                   :ϕ => 1.2,
                   :τ_ϕ => 1.,
                   :bias => 5.,
                   :lapse => 1.,
                   :α => 5.,
                   :k => 10.)
    if remap
       map((x,y)->lb[:x]=y, [:σ2_a, :σ2_i, :σ2_s], repeat([1e-3],3))
       map((x,y)->ub[:x]=y, [:σ2_a, :σ2_i, :σ2_s], repeat([100.],3))
    end
    return lb, ub
end

"""
    is_θlatent_fit_in_jointDDM(;modeltype)

Specify whether each of the parameters controlling the latent variable in the joint drift-diffusion model fit in each type of model

Optional argument:
-`modeltype`: a string specifying the model type. Current options include:
    "nohistory": no influence by trial history
    "history1back": the influence of only the previous trial is included
    "history": the influence of the 30 previous trials is included, and the influence further in the past decays exponentially

Returns:
-`fit` a Dictionary whose values are Bools
"""
function is_θlatent_fit_in_jointDDM(;modeltype::Symbol=:history1back)
    isfit = Dict(  :nohistory => Dict(),
                    :history1back => Dict(),
                    :history => Dict())
    @assert in(modeltype, collect(keys(isfit)))
    isfit[:nohistory] = Dict(   :σ2_i => true,
                                :σ2_a => true,
                                :σ2_s => true,
                                :B => true,
                                :λ => true,
                                :ϕ => true,
                                :τ_ϕ => true,
                                :bias => true,
                                :lapse => true,
                                :α => false,
                                :k => false)
    isfit[:history1back] = Dict(:σ2_i => false,
                                :σ2_a => true,
                                :σ2_s => true,
                                :B => true,
                                :λ => true,
                                :ϕ => true,
                                :τ_ϕ => true,
                                :bias => true,
                                :lapse => true,
                                :α => true,
                                :k => false)
    isfit[:history] = Dict( :σ2_i => false,
                            :σ2_a => true,
                            :σ2_s => true,
                            :B => true,
                            :λ => true,
                            :ϕ => true,
                            :τ_ϕ => true,
                            :bias => true,
                            :lapse => true,
                            :α => true,
                            :k => true)
    return isfit[modeltype]
end

"""
    get_jointDDM_θlatent_names()

Returns an array of Symbols that are the names of the parameters controlling the drift-diffusion process in the joint drift-diffusion model (see ['jointDDM'](@ref)). The names of the parameters specifying the relationship between the firing rate and the latent are not included.

Arguments:

- `θ`: an instance of ['θjoint'](@ref)

Returns:

- param_names: a Vector of Symbols
"""
function get_jointDDM_θlatent_names()
    vcat(collect(fieldnames(typeof(θz()))), collect(fieldnames(typeof(θh()))), map(x->Symbol(x), ["bias", "lapse"]))
end

"""
    specify_a_to_firing_rate_function_type(data)

Specifies the type of the mapping the latent variable (a) to the firing rate for each neuron in each trial-set

Arguments:
- data: A vector of instances of [`jointdata`](@ref)

Optional arguments:
- f: a string indicating the mapping to be either "Softplus" (default) or "Sigmoid"

Returns:
- `ftype`: vector of vector of the string "Softplus" or "Sigmoid"
"""
function specify_a_to_firing_rate_function_type(data::Vector{T}; ftype::String = "Softplus") where {T <: jointdata}

    @assert ftype=="Softplus" || ftype=="Sigmoid"
    ncells = map(x->x.neural_data[1].ncells, data)
    f = repeat([ftype], sum(ncells))
    borg = vcat(0,cumsum(ncells))
    f = [f[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]
end

"""
Constructor method for ([`θjoint`](@ref)) from a vector of parameter values and an array of String arrays specifying the latent-to-neural transformation

Arguments:

- `x` The values of the model parameters
- `f` An array of String arrays specifying the latent-to-neural transformation
"""
function θjoint(x::Vector{T}, f::Vector{Vector{String}}) where {T <: Real}

    @assert all(map(x->x=="Softplus" || x=="Sigmoid", vcat(f...)))
    nparams, ncells = nθparams(f)
    n = count_parameters_in_joint_DDM()
    nh = count_parameters_in_joint_DDM(type=:history)
    nz  = count_parameters_in_joint_DDM(type=:z)

    borg = vcat(n, n .+cumsum(nparams))
    blah = [x[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]

    blah = map((f,x) -> f(x...), getfield.(Ref(@__MODULE__), Symbol.(vcat(f...))), blah)

    borg = vcat(0,cumsum(ncells))
    θy = [blah[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]

    θjoint(θz(x[1:nz]...), θh(x[nz+1:nz+nh]...), x[nz+nh+1], x[nz+nh+2], θy, f)
end

"""
    count_parameters_in_joint_DDM(;,type::String="DDM")

Count the number of parameters in the joint model. The parameters specifying the mapping from the latent to the firing rates are not counted.

Keyword arguments:

- `type` a case-insensitive String specifying the type of parameters to be counted:
    "DDM": all parameters specifying the DDM, including lapse and bias
    "history" or "h": parameters
    "z": paramters specifying the DDM excluding those related to trial history, bias, and lapse

"""
function count_parameters_in_joint_DDM(;type::Symbol=:DDM)
    if type == :DDM
        n = length(fieldnames(typeof(θz()))) +  length(fieldnames(typeof(θh()))) + 2
    elseif type == :history || type == :h
        n = length(fieldnames(typeof(θh())))
    elseif type == :z
        n = length(fieldnames(typeof(θz())))
    else
        error(String(type), " is not recognized")
    end
end

"""
    flatten(θ)

Extract parameters of a [`jointDDM`](@ref) from an instance of [`θjoint`](@ref) and returns an ordered vector.

Arguments:

- `θ`: an instance of ['θjoint'](@ref)

Returns:

- a vector of Floats
```
"""
function flatten(θ::θjoint)

    @unpack θz, θh, bias, lapse, θy = θ
    @unpack σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ = θz
    @unpack α, k = θh
    vcat(σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ, α, k, bias, lapse,
        vcat(collect.(Flatten.flatten.(vcat(θy...)))...))
end

"""
    optimize_jointmodel(model, options)

Optimize model parameters for a ([`jointDDM`](@ref)) using neural and choice data.

Arguments:

- `model`: an instance of ([`jointDDM`](@ref)), a module-defined type that contains the data and parameters from the fit (as well as a few other things that are necessary for re-computing things the way they were computed here (e.g. `n`)
- `options`: ([`joint_options`](@ref)) specifications of the model, including which parameters were fit, their upper and lower bounds, and their initial values.


Returns:

- `model`: an instance of ([`jointDDM`])(@ref)
- `output`: results from [`Optim.optimize`](@ref).

"""
function optimize_jointmodel(model::jointDDM, options::joint_options;
        x_tol::Float64=1e-10, f_tol::Float64=1e-9, g_tol::Float64=1e-3,
        iterations::Int=Int(1e2), show_trace::Bool=true, outer_iterations::Int=Int(1e1),
        scaled::Bool=false, extended_trace::Bool=false, remap::Bool=false)

    @unpack fit, lb, ub = options
    @unpack θ, joint_data, n, cross = model
    @unpack f = θ

    x0 = pulse_input_DDM.flatten(θ)
    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)

    ℓℓ(x) = -joint_loglikelihood(stack(x,c,fit), model; remap=remap)

    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations, scaled=scaled,
        extended_trace=extended_trace)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)

    model = jointDDM(θjoint(x, f), joint_data, n, cross)
    converged = Optim.converged(output)

    return model, output

end

"""
    joint_loglikelihood(x, model)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input. Used
in optimization, Hessian and gradient computation.

Arguments:

- `x`: a vector of model parameters
- 'model': an instance of (['jointDDM'](@ref))

Optional arguments:

- `remap`: boolean indicating whether to compute in the space where the noise terms (σ2_a, σ2_i, σ2_s) are squared

Returns:

- log[P(choices, firing rates|θ, pulses, previous choices/outcomes)] summed across all trials and sessions
"""
function joint_loglikelihood(x::Vector{T}, model::jointDDM; remap::Bool=false) where {T <: Real}
    @unpack joint_data, θ, n, cross = model
    @unpack f = θ
    if remap
        model = jointDDM(θ2(θjoint(x, f)), joint_data, n, cross)
    else
        model = jointDDM(θjoint(x, f), joint_data, n, cross)
    end
    joint_loglikelihood(model)
end

"""
    joint_loglikelihood(model)

Given parameters θ and data (inputs and choices) computes the LL for all trials
"""
joint_loglikelihood(model::jointDDM) = sum(log.(vcat(vcat(joint_likelihood(model)...)...)))

"""
    joint_likelihood(model)

Arguments: (['jointDDM'](@ref)) instance

Returns: `array` of `array` of log[P(choices, firing rates|θ, pulses, previous choices/outcomes)]
"""
function joint_likelihood(model::jointDDM)

    @unpack joint_data, θ, n, cross = model
    @unpack θz, θh, θy, bias, lapse = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    shifted= map(x->x.shifted, joint_data)
    neural_data= map(x->x.neural_data, joint_data)
    @unpack dt = neural_data[1][1].input_data

    P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt) # P is not used

    a₀ = map(x->history_influence_on_initial_point(θh, B, x), shifted)

    map((trialsetdata, θy, trialseta₀) ->
        pmap((trialdata,triala₀) -> joint_likelihood(θ,θy,trialdata,M,xc,dx,n,cross,triala₀),
            trialsetdata, trialseta₀),
        neural_data, θy, a₀)
end

"""
    joint_likelihood(θ,θy,neural_data,M,xc,dx,n,cross,a₀)

Computes the likelihood of the choice and firing rates on a single trial

Arguments:

- `θ`: vector of model parameters
- `θy`: vector of model parameters specifying the mapping from the latent variable to firing rates for the groups of neurons from a single trial-set
- `data`: An instance of ([`neuraldata`](@ref)), containing information on the choice, stimuli, and firing rates on a single trial
- `M`: The transition matrix of P(a_t | a_{t-1}). An `n`-by-`n` matrix, where `n` is the number of bins in the space of the latent variable (a). See [`transition_M`](@ref), [`transition_M!`](@ref)
- `xc`: center of bins in the space of the latent variable (a)
- `dx`: width of bins in the space of the latent variable (a)
- `n`: the number of bins in the space of the latent variable (a)
- `cross`: Bool specifying whether to perform cross-stream click adaptation
- `a₀`: The value of the latent variable at time = 0 (i.e., the time when the stereo-click occured) of a single trial

Returns:

- `LL` the loglikelihood of the firing rates and choice given the stimuli and the choice and outcomes of the previous trials

"""
function joint_likelihood(θ::θjoint, θy, neural_data::neuraldata, M::Matrix{T1},
        xc::Vector{T1}, dx::T1, n::T2, cross::Bool, a₀::T1) where {T1 <: Real, T2 <: Integer}

    @unpack choice = neural_data
    @unpack θz, bias, lapse = θ
    c, P = likelihood(θz, θy, neural_data, M, xc, dx, n, cross, a₀)
    return vcat(c, sum(choice_likelihood!(bias,xc,P,choice,n,dx)) * (1 - lapse) + lapse/2)
end

"""
    likelihood(θz,θy,neural_data,M,xc,dx,n,cross,a₀)

Computes the likelihood of the firing rates from a single trial. See also (θ::Softplus)(x, λ0)

Arguments:

- `θz`: vector of of a subset of model parameters (σ2_i, σ2_a, σ2_s, λ, ϕ, τ_ϕ)
- `θy`: vector of model parameters specifying the mapping from the latent variable to firing rates for the groups of neurons from a single trial-set
- `neural_data`: An instance of ([`neuraldata`](@ref)), containing information on the choice, stimuli, and firing rates on a single trial
- `M`: The transition matrix of P(a_t | a_{t-1}). An `n`-by-`n` matrix, where `n` is the number of bins in the space of the latent variable (a). See [`transition_M`](@ref), [`transition_M!`](@ref)
- `xc`: center of bins in the space of the latent variable (a)
- `dx`: width of bins in the space of the latent variable (a)
- `n`: the number of bins in the space of the latent variable (a)
- `cross`: Bool specifying whether to perform cross-stream click adaptation
- `a₀`: The value of the latent variable at time = 0 (i.e., the time when the stereo-click occured) of a single trial

"""
function likelihood(θz::θz, θy, neural_data::neuraldata, M::Matrix{T1},
        xc::Vector{T1}, dx::T1, n::T2, cross::Bool, a₀::T1) where {T1 <: Real,T2 <: Integer}

    @unpack λ, σ2_a, σ2_i, σ2_s, ϕ, τ_ϕ = θz
    @unpack spikes, input_data = neural_data
    @unpack binned_clicks, clicks, dt, λ0, centered, delay, pad = input_data
    @unpack nT, nL, nR = binned_clicks
    @unpack L, R = clicks
    @unpack dt = neural_data.input_data

    #adapt magnitude of the click inputs
    La, Ra = adapt_clicks(ϕ,τ_ϕ,L,R;cross=cross)

    F = zeros(T1,n,n) #empty transition matrix for time bins with clicks

    time_bin = (-(pad-1):nT+pad) .- delay

    c = Vector{T1}(undef, length(time_bin))

    P = P0(σ2_i, a₀, n, dx, xc, dt)

    @inbounds for t = 1:length(time_bin)

        if time_bin[t] >= 1
            P, F = latent_one_step!(P, F, λ, σ2_a, σ2_s, time_bin[t], nL, nR, La, Ra, M, dx, xc, n, dt)
        end

        #weird that this wasn't working....
        #P .*= vcat(map(xc-> exp(sum(map((k,θy,λ0)-> logpdf(Poisson(θy(xc,λ0[t]) * dt),
        #                        k[t]), spikes, θy, λ0))), xc)...)

        P = P .* (vcat(map(xc->
                exp(sum(map((k,θy,λ0)->
                    logpdf(Poisson(θy(xc,λ0[t]) * dt), k[t]),
                    spikes, θy, λ0))),
                xc)...))
        c[t] = sum(P)
        P /= c[t]
    end

    return c, P
end

"""
    history_influence_on_initial_point(θh, data, B)

Computes the influence of trial history on the initial point for all the trials of a trial-set

Arguments:

-`θhist`: trial history parameters (['θh'](@ref))
-'B': bound
-'shifted': (['trialshifted'](@ref))
"""
function history_influence_on_initial_point(θhist::θh, B, shifted::trialshifted)
    @unpack α, k = θhist
    @unpack choice, reward, shift = shifted
    a₀ = sum(α.*choice.*reward.*exp.(k.*(shift.+1)), dims=2)[:,1]
    min.(max.(a₀, -B), B)
end

"""
    choice_optimize(model, options)

Optimize choice-related model parameters for a [`jointDDM`](@ref) using only choice data and no neural data.

Arguments:

- `model`: an instance of a [`jointDDM`](@ref).
- `options`: an instance of [`joint_options`](@ref) specifying the model architecture, data preprocessing, and optimization procedure, such as which parameters were fit (`fit`), and the upper (`ub`) and lower (`lb`) bounds of those parameters.

Returns:

- `model`: an instance of a `neural_choiceDDM`.
- `output`: results from [`Optim.optimize`](@ref).
"""
function choice_optimize(model::jointDDM, options::joint_options;
        x_tol::Float64=1e-10, f_tol::Float64=1e-9, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true, outer_iterations::Int=Int(1e1),
        scaled::Bool=false, extended_trace::Bool=false, remap::Bool=false)

    @unpack fit, lb, ub = options
    @unpack θ, joint_data, n, cross = model
    @unpack f = θ

    x0 = pulse_input_DDM.flatten(θ)
    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)

    ℓℓ(x) = -choice_loglikelihood(stack(x,c,fit), model; remap=remap)

    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations, scaled=scaled,
        extended_trace=extended_trace)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)

    model = jointDDM(θjoint(x, f), joint_data, n, cross)
    converged = Optim.converged(output)

    return model, output
end

"""
    choice_loglikelihood(x, model)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input. Used
in optimization, Hessian and gradient computation.
"""
function choice_loglikelihood(x::Vector{T}, model::jointDDM; remap::Bool=false) where {T <: Real}

    @unpack joint_data,θ,n,cross = model
    @unpack f = θ
    if remap
        model = jointDDM(θ2(θjoint(x, f)), joint_data, n, cross)
    else
        model = jointDDM(θjoint(x, f), joint_data, n, cross)
    end
    choice_loglikelihood(model)
end

"""
    choice_loglikelihood(model)

Given parameters θ and data (sound pulses and choices) computes the LL for all trial-sets
"""
choice_loglikelihood(model::jointDDM) = sum(log.(vcat(choice_likelihood(model)...)))

"""
    choice_likelihood(model)

Arguments: `jointDDM` instance

Returns: `array` of `array` of P(choices|θ, sound pulses)
"""
function choice_likelihood(model::jointDDM)

    @unpack joint_data,θ,n,cross = model
    @unpack θz, θh, bias, lapse = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    neural_data = map(x->x.neural_data, joint_data)
    shifted = map(x->x.shifted, joint_data)
    choice = map(x->map(x->x.choice), neural_data)
    @unpack dt = neural_data[1][1].input_data
    click_data = map(x->map(x->x.input_data.clicks), neural_data)

    a₀ = map(x->history_influence_on_initial_point(θh, B, x), shifted)

    P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt) # P is not used

    map((choice_trialset, clicks_trialset, a₀_trialset) ->
        pmap((choice_trial, clicks_trial, a₀_trial) ->
                choice_likelihood(θ,choice_trial,clicks_trial,dt,M,xc,dx,n,cross,a₀_trial),
            choice_trialset, clicks_trialset, a₀_trialset),
        choice, click_data, a₀)
end

"""
    choice_likelihood(θ,choice,click_data,dt,M,xc,dx,n,cross,a₀)

Computes the likelihood of choice in a single trial

Arguments:

- `θ`: parameters of a joint model and an instance of ['θjoint'](@ref)
- `choice`: A Bool indicating the choice of a trial (0 indicates left)
- `click_data`: the click times of a trial and an instance of [`clicks`](@ref)
- `dt`: duration of each time step in seconds
- `M`: stochastic matrix for mapping P(t-1) to P(t)
- `xc`: center of the bins in the space of the latent variable (a)
- `dx`: width of the bins in a-space
- `n`: number of bins in a-space
- `cross`:true indicates cross-stream, rather than within-stream, adaptation
- `a₀`: value of the latent variable at the first time point of the trial
"""
function choice_likelihood(θ::θjoint, choice::Bool, click_data::clicks, dt::T1, M::Matrix{T1}, xc::Vector{T1}, dx::T1, n::T2, cross::Bool, a₀::T1) where {T1<:Real, T2<:Integer}

    @unpack θz, bias, lapse = θ
    @unpack σ2_i = θz

    P = P0(σ2_i, a₀, n, dx, xc, dt)
    P = P_single_trial!(θz,P,M,dx,xc,click_data,n,cross)
    sum(choice_likelihood!(bias,xc,P,choice,n,dx)) * (1 - lapse) + lapse/2
end

"""
    Hessian(model; chunck_size, remap)

Compute the hessian of the negative log-likelihood at the current value of the parameters of a `jointDDM`.

Arguments:

- `model`: instance of [`jointDDM`](@ref)

Optional arguments:

- `chunk_size`: parameter to manange how many passes over the LL are required to compute the Hessian. Can be larger if you have access to more memory.
- `remap`: For considering parameters in variance of std space.

"""
function Hessian(model::jointDDM; chunk_size::Int=4, remap::Bool=false)

    @unpack θ = model
    x = flatten(θ)
    ℓℓ(x) = -joint_loglikelihood(x, model; remap=remap)

    cfg = ForwardDiff.HessianConfig(ℓℓ, x, ForwardDiff.Chunk{chunk_size}())
    ForwardDiff.hessian(ℓℓ, x, cfg)
end

"""
    confidence_interval(H, θ; confidence_interval)

Estimate the confidence level of the coefficients using the Hessian

Arugments:

-`H`: A Hessian matrix whose elements are Floats
-`θ`: an instance of a type that is a subtype of `DDMθ`, such as ['θjoint'](@ref)

Optional argument:

-`confidence_level`: a Real in the interval [0,100]. Default is 95

Returns:

-A matrix with two columns representing the lower and upper bounds
"""
function confidence_interval(H::Matrix{T1}, θ::θjoint; confidence_level::T2 = 95.) where {T1 <:Real, T2 <:Real}
    confidence_level = convert(Float64, confidence_level)
    @assert confidence_level >= 0. && confidence_level <= 100.
    if det(H) == 0.0
        repeat([NaN], size(H)[1],2)
    else
        σ2 = diag(inv(H))
        σ2[σ2 .< 0.] .= NaN
        σ = σ2.^0.5
        z = quantile(Normal(), (1. + confidence_level/100)/2)
        flatten(θ) .+ z.*hcat(-σ, σ)
    end
end
"""
    θ2(θ)

Square the values of a subset of parameters (σ2_i,σ2_a, σ2_s)
"""
θ2(θ::θjoint) = θjoint(θz=θz2(θ.θz), bias=θ.bias, lapse=θ.lapse, θy=θ.θy, f=θ.f)

"""
    invθ2(θ)

Returns the positive square root of a subset of parameters (σ2_i,σ2_a, σ2_s)
"""
invθ2(θ::θjoint) = θjoint(θz=invθz2(θ.θz), bias=θ.bias, lapse=θ.lapse, θy=θ.θy, f=θ.f)



"""
    P_goright(model)

Given an instance of [`jointDDM`](@ref) computes the probabilty of going right for each trial.
"""
function P_goright(model::jointDDM)

    @unpack θ, joint_data, n, cross = model
    @unpack θz, θy, bias, lapse = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    neural_data = map(x->x.neural_data, joint_data)
    shifted = map(x->x.shifted, joint_data)
    click_data = map(x->map(x->x.input_data.clicks), neural_data)
    @unpack dt = neural_data[1][1].input_data

    a₀ = map(x->history_influence_on_initial_point(θh, B, x), shifted)

    P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt) # P is not used

    map((a₀_trialset, clicks_trialset) ->
            pmap((a₀_trial, clicks_trial) ->
                    P_goright(θ,click_data,dt,M,xc,dx,n,cross,a₀_1trial),
                a₀_trialset, clicks_trialset),
        a₀, click_data)

end


"""
    P_goright(θ,click_data,dt,M,xc,dx,n,cross,a₀)

Computes the likelihood of a right choice for an individual trial

Arguments:

- `θ`: parameters of a joint model and an instance of ['θjoint'](@ref)
- `click_data`: the click times of a trial and an instance of [`clicks`](@ref)
- `dt`: duration of each time step in seconds
- `M`: stochastic matrix for mapping P(t-1) to P(t)
- `xc`: center of the bins in the space of the latent variable (a)
- `dx`: width of the bins in a-space
- `n`: number of bins in a-space
- `cross`:true indicates cross-stream, rather than within-stream, adaptation
- `a₀`: value of the latent variable at the first time point of the trial
"""
function P_goright(θ::θjoint, click_data::clicks, dt::T1, M::Matrix{T1}, xc::Vector{T1}, dx::T1, n::T2, cross::Bool, a₀::T1) where {T1<:Real, T2<:Integer}

    @unpack θz, bias, lapse = θ

    P = P0(σ2_i, a₀, n, dx, xc, dt)
    P = P_single_trial!(θz,P,M,dx,xc,click_data,n,cross)
    sum(choice_likelihood!(bias,xc,P,true,n,dx)) * (1 - lapse) + lapse/2
end


"""
    gradient(model; remap)

Compute the gradient of the negative log-likelihood at the current value of the parameters of a [`jointDDM`](@ref).

Arguments:

- `model`: instance of [`jointDDM`](@ref)

Optional arguments:

- `remap`: For considering noise parameters (σ2_i, σ2_a, σ2_s) in variance of std space.
"""
function gradient(model::jointDDM; remap::Bool=false)

    @unpack θ = model
    x = flatten(θ)
    ℓℓ(x) = -joint_loglikelihood(x, model; remap=remap)

    ForwardDiff.gradient(ℓℓ, x)
end
