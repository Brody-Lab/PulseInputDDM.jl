"""
Arguments:

- `f`: function mapping the latent variable to firing rate. It can be either "Softplus" or "Sigmoid"

Optional arguments:

- `remap`: if true, parameters are considered in variance instead of std space
- `modeltype`: a string specifying the model type. Current options include:
    "nohistory"
    "history1back" the influence of only the previous trial is included
    "history" the influence of the 30 previous trials is included, and the influence further in the past decays exponentially

Returns:

- ([`joint_options`](@ref))
"""
function specify_jointmodel(f; remap::Bool=false, modeltype = "history1back")
    θDDM_lb = Dict( :σ2_i => 0.,
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
    θDDM_ub = Dict( :σ2_i => 40.,
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
    θDDM_x0 = Dict( :σ2_i => eps(),
                    :σ2_a => eps(),
                    :σ2_s => eps(),
                    :B => 40.,
                    :λ => 0.,
                    :ϕ => 1.,
                    :τ_ϕ => 0.1,
                    :bias => 0.,
                    :lapse => eps(),
                    :α => 0.,
                    :k => eps())
    if remap
        map((x,y)->θDDM_lb[:x]=y, [:σ2_a, :σ2_i, :σ2_s], repeat([1e-3],3))
        map((x,y)->θDDM_ub[:x]=y, [:σ2_a, :σ2_i, :σ2_s], repeat([100.],3))
        map((x,y)->θDDM_x0[:x]=y, [:σ2_a, :σ2_i, :σ2_s], repeat([1.],3))
    end
    is_fit = Dict(  :nohistory => Dict(),
                    :history1back => Dict(),
                    :history => Dict())
    is_fit[:nohistory] = Dict(  :σ2_i => true,
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
    is_fit[:history1back] = Dict(   :σ2_i => false,
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
    is_fit[:history] = Dict(:σ2_i => false,
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
    modeltype = Symbol(modeltype)
    assert(in(modeltype, collect(keys(is_fit))))
    paramnames = get_DDM_param_names(θjoint())
    ub = Vector{Float64}(undef,length(paramnames))
    lb = Vector{Float64}(undef,length(paramnames))
    fit = Vector{Bool}(undef,length(paramnames))
    x0 = Vector{Float64}(undef,length(paramnames))
    for i in 1:length(paramnames)
        lb[i] = θDDM_lb[paramnames[i]]
        ub[i] = θDDM_ub[paramnames[i]]
        x0[i] = θDDM_x0[paramnames[i]]
        fit[i] = is_fit[modeltype][paramnames[i]]
        if fit[i]
            x0[i] = lb[i] + (ub[i] - lb[i]) * rand()
        end
    end

    n_neural_params, ncells = nθparams(f)
    fit = vcat(fit, trues.(n_neural_params)...)

    for i in 1:sum(ncells)
        if vcat(f...)[i] == "Softplus"
            lb = vcat(lb,-10.)
            ub = vcat(ub, 10.)
            x0 = vcat(x0, 0.)
        elseif vcat(f...)[i] == "Sigmoid"
            lb = vcat(lb, [-100.,0.,-10.,-10.])
            ub = vcat(ub, [100.,100.,10.,10.])
            x0 = vcat(x0, [0.,1.,0.,0.])
        end
    end
    joint_options(lb = lb, ub = ub, fit = fit, x0 = x0)
end

"""
    get_DDM_parameter_names(θ)

Returns an array of Symbols that are the names of the parameters controlling the drift-diffusion process in the joint drift-diffusion model (see ['jointDDM'](@ref)). The names of the parameters specifying the relationship between the firing rate and the latent are not included.

Arguments:

- `θ`: an instance of ['θjoint'](@ref)

Returns:

- param_names: a Vector of Symbols
"""
function get_DDM_parameter_names(θ::θjoint)
    params = vcat(collect(fieldnames(typeof(θ.θz))),
                  collect(fieldnames(typeof(θ.θh))),
                  map(x->Symbol(x), ["bias", "lapse"]))
end

"""
Constructor method for ([`θjoint`](@ref)) from a vector of parameter values and an array of String arrays specifying the latent-to-neural transformation

Arguments:

- `x` The values of the model parameters
- `f` An array of String arrays specifying the latent-to-neural transformation
"""
function θjoint(x::Vector{T}, f::Vector{Vector{String}}) where {T <: Real}

    nparams, ncells = nθparams(f)
    n = count_parameters_in_joint_DDM
    nh = count_parameters_in_joint_DDM(type=:history)
    nz  = count_parameters_in_joint_DDM(type=:z)

    borg = vcat(n, n .+cumsum(nparams))
    blah = [x[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]

    blah = map((f,x) -> f(x...), getfield.(Ref(@__MODULE__), Symbol.(vcat(f...))), blah)

    borg = vcat(0,cumsum(ncells))
    θy = [blah[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]

    θjoint(θz(x[1:nz]...), θh(x[nz+1:nz+nh]) x[nz+nh+1], x[nz+nh+2], θy, f)
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
    type = lowercase(type)
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
    @unpack α, β = θh
    vcat(σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ, α, β, bias, lapse,
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
        iterations::Int=Int(2e3), show_trace::Bool=true, outer_iterations::Int=Int(1e1),
        scaled::Bool=false, extended_trace::Bool=false, remap::Bool=false)

    @unpack fit, lb, ub = options
    @unpack θ, data, n, cross = model
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

    model = jointDDM(θjoint(x, f), data, n, cross)
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
function joint_loglikelihood(x::Vector{T}, model::jointDDM; remap::Bool=false) where {T <: AbstractFloat}
    @unpack data,θ,n,cross = model
    @unpack f = θ
    if remap
        model = jointDDM(θ2(θjoint(x, f)), data, n, cross)
    else
        model = jointDMM(θjoint(x, f), data, n, cross)
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
    @unpack shifted, neural_data = joint_data
    @unpack dt = neural_data[1][1].input_data

    P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt) # P is not used

    a₀ = map(x->history_influence_on_initial_point(θh, B, x), shifted)

    map((trialsetdata, θy, trialseta₀) ->
        pmap(trialdata,triala₀ -> joint_likelihood(θ,θy,trialdata,M,xc,dx,n,cross,triala₀),
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
function joint_likelihood(θ::Vector{T1}, θy::Vector{T1}, neural_data::neuraldata, M::Matrix{T1},
        xc::Vector{T1}, dx::T1, n::T2, cross::Bool, a₀::T1) where {T1 <: AbstractFloat, T2 <: Integer}

    @unpack choice = neural_data
    @unpack θz, bias, lapse = θ
    c, P = likelihood(θz, θy, neural_data, M, xc, dx, n, cross, a₀)
    return vcat(c, sum(choice_likelihood!(bias,xc,P,choice,n,dx)) * (1 - lapse) + lapse/2)
end

"""
    likelihood(θz,θy,neural_data,M,xc,dx,n,cross,a₀)

Computes the likelihood of the firing rates from a single trial

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
function likelihood(θz::θz, θy::Vector{T1}, neural_data::neuraldata, M::Matrix{T1},
        xc::Vector{T1}, dx::T1, n::T2, cross::Bool, a₀::T1) where {T1 <: AbstractFloat,T2 <: Integer}

    @unpack λ, σ2_a, σ2_i, σ2_s, ϕ, τ_ϕ = θz
    @unpack spikes, input_data = neural_data
    @unpack binned_clicks, clicks, dt, λ0, centered, delay, pad = input_data
    @unpack nT, nL, nR = binned_clicks
    @unpack L, R = clicks

    #adapt magnitude of the click inputs
    La, Ra = adapt_clicks(ϕ,τ_ϕ,L,R;cross=cross)

    F = zeros(T1,n,n) #empty transition matrix for time bins with clicks

    time_bin = (-(pad-1):nT+pad) .- delay

    c = Vector{T1}(undef, length(time_bin))

    P = P0(σ2_i, a₀, n, dx, xc, click_data.dt)

    @inbounds for t = 1:length(time_bin)

        if time_bin[t] >= 1
            P, F = latent_one_step!(P, F, λ, σ2_a, σ2_s, time_bin[t], nL, nR, La, Ra, M, dx, xc, n, dt)
        end

        #weird that this wasn't working....
        #P .*= vcat(map(xc-> exp(sum(map((k,θy,λ0)-> logpdf(Poisson(θy(xc,λ0[t]) * dt),
        #                        k[t]), spikes, θy, λ0))), xc)...)

        P = P .* (vcat(map(xc-> exp(sum(map((k,θy,λ0)-> logpdf(Poisson(θy(xc,λ0[t]) * dt),
                        k[t]), spikes, θy, λ0))), xc)...))
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

    model = jointDDM(θjoint(x, f), data, n, cross)
    converged = Optim.converged(output)

    return model, output
end

"""
    choice_loglikelihood(x, model)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input. Used
in optimization, Hessian and gradient computation.
"""
function choice_loglikelihood(x::Vector{T}, model::jointDDM; remap::Bool=false) where {T <: AbstractFloat}

    @unpack joint_data,θ,n,cross = model
    @unpack f = θ
    if remap
        model = jointDDM(θ2(θjoint(x, f)), data, n, cross)
    else
        model = jointDDM(θjoint(x, f), data, n, cross)
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
        pmap(choice_trial, clicks_trial, a₀_trial ->
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
function choice_likelihood(θ::θjoint, choice::Bool, click_data::clicks, dt::T1, M::Matrix{T1}, xc::Vector{T1}, dx::T1, n::T3, cross::Bool, a₀::T1) where {T1<:AbstractFloat, T2<:Integer}

    @unpack θz, bias, lapse = θ
    @unpack σ2_i = θz

    P = P0(σ2_i, a₀, n, dx, xc, dt)
    P = P_single_trial!(θz,P,M,dx,xc,click_data,n,cross)
    sum(choice_likelihood!(bias,xc,P,choice,n,dx)) * (1 - lapse) + lapse/2
end

"""
    Hessian(model; chunck_size, remap)

Compute the hessian of the negative log-likelihood at the current value of the parameters of a `joint_DDM`.

Arguments:

- `model`: instance of [`joint_DDM`](@ref)

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

Given an instance of `joint_DDM` computes the probabilty of going right for each trial.
"""
function P_goright(model::joint_DDM)

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
            pmap(a₀_trial, clicks_trial ->
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
function P_goright(θ::θjoint, click_data::clicks, dt::T1, M::Matrix{T1}, xc::Vector{T1}, dx::T1, n::T2, cross::Bool, a₀::T1) where {T1<:AbstractFloat, T2<:Integer}

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

"""
    *** This is not currently used***

    settings

Additional parameters for processing data the data and fitting the model.

Fields:

- `break_sim_data`: this will break up simulatenously recorded neurons, as if they were recorded independently. Not often used by most users.
- `centered`: Defaults to true. For the neural model, this aligns the center of the binned spikes, to the beginning of the binned clicks. This was done to fix a numerical problem. Most users will never need to adjust this.
- `cut`: How much extra to cut off at the beginning and end of filtered things (should be equal to `extra_pad` in most cases).
- `delay`: How much to offset the spikes, relative to the accumlator, in units of `dt`.
- `dt`: Binning of the spikes, in seconds.
- `extra_pad`: Extra padding (in addition to `pad`) to add, for filtering purposes. In units of `dt`.
- `filtSD`: standard deviation of a Gaussin (in units of `dt`) to filter the spikes with to generate single trial firing rates (`μ_rnt`), and mean firing rate across all trials (`μ_t`).
- `pad`: How much extra time should spikes be considered before and after the begining of the clicks. Useful especially if delay is large.
- `pcut`: p-value for selecting cells.
"""
@with_kw struct settings {T1::Bool, T2<:Integer, T3<:AbstractFloat}
    break_sim_data::T1=false,
    centered::T1=true
    cut::T2=10
    delay::T2=0
    do_RBF::T1=false
    dt::T3=1e-2
    extra_pad::T2=10
    filtSD::T2=2,
    nRBFs::T2=6
    pad::T2=0
    pcut::T3=0.01,
end
