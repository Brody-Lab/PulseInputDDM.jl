"""
    choiceoptions(fit, lb, ub)

Fields:

- `fit`: `array` of `Bool` for optimization for `choiceDDM` model.
- `lb`: `array` of lower bounds for optimization for `choiceDDM` model.
- `ub`: `array` of upper bounds for optimization for `choiceDDM` model.

also see function `create_options_and_x0`
"""
@with_kw mutable struct choiceoptions
    fit::Vector{Bool} 
    lb::Vector{Float64}  
    ub::Vector{Float64} 
end


"""
     create_options_and_x0(;modeltype)

Creates options and initial value for choiceDDM.

Returns:

- `options`: an instance of `choiceoptions`: module-defind type that contains 
            the upper (`ub`) and lower (`lb`) boundaries and specification of 
            which parameters to fit (`fit`).
- `x0`: initial value of params for fitting

Keyword Arguments:

- `modeltype`: could be `bing`, `hist_initpt`, `hist_initpt_lapse`, `hist_lapse`
               returns `options.fit` and `x0` to be be consistent with 
               these classes of model 
               if fitting `hist_initpt_*` set `initpt_mod` to `true` while optimizing
               default is `bing`

"""
function create_options_and_x0(; modeltype = "bing")

     paramlims = Dict( 
      #:paramname => [lb, ub, fit_bing, fit_hist_initpt, hist_initpt_lapse,hist_lapse,  nofit_default]
        :σ2_i =>        [0., 40., true, false, false, true, eps()], 
        :B =>           [5., 60., true, true, true, true, 40.],      
        :λ =>           [-5., 5., true, true, true, true, 1. + eps()],                                            
        :σ2_a =>        [0., 100., true, true, true, true, eps()], 
        :σ2_s =>        [0., 8., true, true, true, true, eps()],  
        :ϕ =>           [0.01, 1.2, true, true, true, true, 1. + eps()], 
        :τ_ϕ =>         [0.005, 1., true, true, true, true, eps()],   
        :lapse_prob =>  [0., 0.5, true, true, true, true, eps()],                  
        :lapse_bias =>  [-5., 5., false, true, true, true, 0.], 
        :lapse_modbeta=>[0., 2., false, false, true, true, 0.],                                 
        :h_ηc =>       [-3.5, 3.5, false, true, true, true, 0.], 
        :h_ηe =>       [-3.5, 3.5, false, true, true, true, 0.], 
        :h_βc =>        [0., 1., false, true, true, true, 0.], 
        :h_βe =>        [0., 1., false, true, true, true, 0.],
        :bias =>        [-5., 5., true, true, true, true, 0.])        

    modeltype_idx = Dict(
        "bing"              => 3,
        "hist_initpt"       => 4,
        "hist_initpt_lapse" => 5,
        "hist_lapse"        => 6)

    params = get_param_names(θchoice())

    ub = Vector{Float64}(undef,length(params))
    lb = Vector{Float64}(undef,length(params))
    fit = Vector{Bool}(undef,length(params))
    x0 = Vector{Float64}(undef,length(params))

    for i in 1:length(params)
        lb[i] = paramlims[Symbol(params[i])][1]
        ub[i] = paramlims[Symbol(params[i])][2]
        fit[i] = paramlims[Symbol(params[i])][modeltype_idx[modeltype]]
        if fit[i]
            x0[i] = lb[i] + (ub[i] - lb[i]) * rand()
        else
            x0[i] = paramlims[Symbol(params[i])][7]
        end
    end

    options = choiceoptions(lb = lb, ub = ub, fit = fit)
    return options, x0

end


"""
    θchoice(θz, θlapse, θhist, bias) <: DDMθ

Fields:

- `θz`: is a module-defined type that contains the parameters related to the latent variable model.
- `θlapse` : is a module-defined type that contains the parameters related to lapse probability
- `θhist` : is a module-defined type that contains the parameters that determine effect of trial-history
- `bias` is the choice bias parameter.

Example:

```julia
θchoice(
    θz=θz(σ2_i = 0.5, B = 15., λ = -0.5, σ2_a = 50., σ2_s = 1.5, ϕ = 0.8, τ_ϕ = 0.05), 
    bias=1., 
    θlapse=θlapse(lapse_prob = 0.05, lapse_bias = 0.5, lapse_modbeta = 2.),
    θhist=θtrialhist(h_ηc = 0.3, h_ηe = -0.1, h_βc = 0.9, h_βe = 0.1))
```
"""
@with_kw struct θchoice{T1, T2, T3, T<:Real} <: DDMθ
    θz::T1 = θz()
    θlapse::T2 = θlapse()
    θhist::T3 = θtrialhist()
    bias::T = 1.
end


"""
get_param_names(θ)
given θ, returns an array with all the param names as strings
currently written only for θchoice but should be extended to other types
"""
function get_param_names(θ::θchoice)
    params = vcat(collect(map(x-> string(x), fieldnames(typeof(θ.θz)))), 
            collect(map(x-> string(x), fieldnames(typeof(θ.θlapse)))),
            collect(map(x-> string(x), fieldnames(typeof(θ.θhist)))),
            collect(map(x-> string(x), fieldnames(typeof(θ)))[4:end]))  
end





"""
    choicedata{T1} <: DDMdata

Fields:

- `click_data` is a type that contains all of the parameters related to click input.
- `choice` is the choice data for a single trial.

Example:

```julia
```
"""
@with_kw struct choicedata{T1} <: DDMdata
    click_data::T1
    choice::Bool
end


"""
    choiceDDM(θ, data, n, cross, initpt_mod)

Fields:

- `θ`: a instance of the module-defined class `θchoice` that contains all of the model parameters for a `choiceDDM`
- `data`: an `array` where each entry is the module-defined class `choicedata`, which contains all of the data (inputs and choices).
- `n`: number of spatial bins to use (defaults to 53).
- `cross`: whether or not to use cross click adaptation (defaults to false).
- `initpt_mod`: whether or not to modulate initial point with trial history

Example:

```julia
ntrials, dt, centered, n  = 1, 1e-2, false, 53
θ = θchoice()
_, data = synthetic_data(n ;θ=θ, ntrials=ntrials, rng=1, dt=dt);
choiceDDM(θ=θ, data=data, n=n)
```
"""
@with_kw struct choiceDDM{T,U,V} <: DDM
    θ::T = θchoice()
    data::U
    dx::Float64=0.25
    cross::Bool=true
    initpt_mod::Bool=false
    θprior::V = θprior()
end


"""
"""
function train_and_test(data, options::choiceoptions; 
        dx::Float64=0.25, cross::Bool=true, initpt_mod=false,
        x_tol::Float64=1e-10, f_tol::Float64=1e-9, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true, outer_iterations::Int=Int(1e1),
        extended_trace::Bool=false, scaled::Bool=false, time_limit::Float64=170000., show_every::Int=10,
        x0::Vector{Float64} = [0.1, 15., -0.1, 20., 0.5, 0.8, 0.008, 0.01, 0., 0., 0., 0., 0., 0., 0.], 
        seed::Int=1, σ_B::Float64=1e6)
        
    ntrials = length(data)
    train = sample(Random.seed!(seed), 1:ntrials, ceil(Int, 0.9 * ntrials), replace=false)
    test = setdiff(1:ntrials, train)
    
    θ = Flatten.reconstruct(θchoice(), x0)
    model = choiceDDM(θ, data[train], dx, cross, initpt_mod, θprior(μ_B=40., σ_B=σ_B))
    
    model, = optimize(model, options; 
        x_tol=x_tol, f_tol=f_tol, g_tol=g_tol, iterations=iterations, show_trace=show_trace, 
        outer_iterations=outer_iterations, extended_trace=extended_trace, 
        scaled=scaled, time_limit=time_limit, show_every=show_every)
        
    testLL = loglikelihood(choiceDDM(model.θ, data[test], dx, cross, initpt_mod, θprior(μ_B=40., σ_B=σ_B)))
    LL = loglikelihood(choiceDDM(model.θ, data, dx, cross, initpt_mod, θprior(μ_B=40., σ_B=σ_B)))

    return σ_B, model, testLL, LL
    
end


"""
    optimize(model, options)

Optimize model parameters for a `choiceDDM`.

Returns:

- `model`: an instance of a `choiceDDM`.
- `output`: results from [`Optim.optimize`](@ref).

Arguments:

- `model`: an instance of a `choiceDDM`.
- `options`: module-defind type that contains the upper (`ub`) and lower (`lb`) boundaries and specification of which parameters to fit (`fit`).

"""
function optimize(model::choiceDDM, options::choiceoptions; 
        x_tol::Float64=1e-10, f_tol::Float64=1e-9, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true, outer_iterations::Int=Int(1e1),
        extended_trace::Bool=false, scaled::Bool=false, time_limit::Float64=170000., show_every::Int=5)

    @unpack fit, lb, ub = options
    @unpack θ, data, dx, cross, initpt_mod, θprior = model
    
    x0 = collect(Flatten.flatten(θ))

    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)

    ℓℓ(x) = -(loglikelihood(stack(x,c,fit), model) + logprior(stack(x,c,fit), θprior))
    
    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations, extended_trace=extended_trace,
        scaled=scaled, time_limit=time_limit, show_every=show_every)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    θ = Flatten.reconstruct(θchoice(), x)
    model = choiceDDM(θ, data, dx, cross, initpt_mod, θprior)
    converged = Optim.converged(output)

    return model, output

end


"""
    optimize(data, options)

Optimize model parameters for a `choiceDDM`.

Returns:

- `model`: an instance of a `choiceDDM`.
- `output`: results from [`Optim.optimize`](@ref).

Arguments:

- `data`: an `array`, each element of which is a module-defined type `choicedata`. `choicedata` contains the click data and the choice for a trial.
- `options`: module-defind type that contains the upper (`ub`) and lower (`lb`) boundaries and specification of which parameters to fit (`fit`).

"""
function optimize(data, options::choiceoptions; 
        dx::Float64=0.25, cross::Bool=true, initpt_mod::Bool=false,
        x_tol::Float64=1e-10, f_tol::Float64=1e-9, g_tol::Float64=1e-6,
        iterations::Int=Int(2e3), show_trace::Bool=true, outer_iterations::Int=Int(1e1),
        extended_trace::Bool=false, scaled::Bool=false, time_limit::Float64=170000., show_every::Int=5,
        x0::Vector{Float64} = [0.1, 15., -0.1, 20., 0.5, 0.8, 0.008, 0.01, 0., 0., 0., 0., 0., 0., 0.],  
        θprior::θprior=θprior())
    
    θ = Flatten.reconstruct(θchoice(), x0)
    model = choiceDDM(θ, data, dx, cross, initpt_mod, θprior)
    
    model, output = optimize(model, options; 
        x_tol=x_tol, f_tol=f_tol, g_tol=g_tol, iterations=iterations, show_trace=show_trace, 
        outer_iterations=outer_iterations, extended_trace=extended_trace, 
        scaled=scaled, time_limit=time_limit, show_every=show_every)

    return model, output

end


"""
    loglikelihood(x, model)

Given a vector of parameters and a type containing the data related to the choice DDM model, compute the LL.

See also: [`loglikelihood`](@ref)
"""
function loglikelihood(x::Vector{T1}, model::choiceDDM) where {T1 <: Real}

    @unpack dx, data, cross, initpt_mod, θprior = model
    θ = Flatten.reconstruct(θchoice(), x)
    model = choiceDDM(θ, data, dx, cross, initpt_mod, θprior)
    loglikelihood(model)

end


"""
    gradient(model)

Compute the gradient of the negative log-likelihood at the current value of the parameters of a `choiceDDM`.
"""
function gradient(model::choiceDDM)

    @unpack θ = model
    x = [Flatten.flatten(θ)...]
    ℓℓ(x) = -loglikelihood(x, model)

    ForwardDiff.gradient(ℓℓ, x)

end


"""
    Hessian(model)

Compute the hessian of the negative log-likelihood at the current value of the parameters of a `choiceDDM`.
"""
function Hessian(model::choiceDDM)

    @unpack θ = model
    x = [Flatten.flatten(θ)...]
    ℓℓ(x) = -loglikelihood(x, model)

    ForwardDiff.hessian(ℓℓ, x)

end



"""
"""
θ2(θ::θchoice) = θchoice(θz=θz2(θ.θz), bias=θ.bias, 
                        θlapse=θ.θlapse, θhist = θ.θhist)   



"""
"""
θexp(θ) = θchoice(θz=θz(σ2_i = exp(θ.θz.σ2_i), B = θ.θz.B, λ = θ.θz.λ, 
        σ2_a = exp(θ.θz.σ2_a), σ2_s = exp(θ.θz.σ2_s), 
        ϕ = θ.θz.ϕ, τ_ϕ = θ.θz.τ_ϕ), 
        bias=θ.bias, θlapse=θ.θlapse, θhist = θ.θhist)   
    
    
"""
    loglikelihood(model)

Given parameters θ and data (inputs and choices) computes the LL for all trials
"""
function loglikelihood(model::choiceDDM)
    
    @unpack θ, data, dx, cross, initpt_mod = model
    @unpack θz, θhist  = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    @unpack dt = data[1].click_data

    i_0 = compute_history(θhist, data, B)
    M,xc,n = initialize_latent_model(σ2_i, B, λ, σ2_a, dx, dt)   

    sum(pmap((data, i_0) -> loglikelihood!(θ, M, dx, xc, data, i_0, n, cross, initpt_mod), data, i_0))

end


"""
    loglikelihood!(θ, P, M, dx, xc, data, n, cross)

Given parameters θ and data (inputs and choices) computes the LL for one trial
"""
loglikelihood!(θ::θchoice, M::Array{TT,2}, dx::Float64, xc::Vector{TT}, 
        data::choicedata, i_0::TT, n::Int, 
        cross::Bool, initpt_mod::Bool) where {TT <: Real} = log(likelihood!(θ, M, dx, xc, data, i_0, n, cross, initpt_mod))


"""
    likelihood(model)

Given parameters θ and data (inputs and choices) computes the likehood of the choice for all trials
"""
function likelihood(model::choiceDDM)
    
    @unpack θ, data, dx, cross, initpt_mod = model
    @unpack θz, θhist = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    @unpack dt = data[1].click_data

    i_0 = compute_history(θhist, data, B)
    M,xc,n = initialize_latent_model(σ2_i, B, λ, σ2_a, dx, dt)

    pmap((data, i_0) -> likelihood!(θ, M, dx, xc, data, i_0, n, cross, initpt_mod), data, i_0)

end


"""
    P_goright(model)

Given an instance of `choiceDDM` computes the probabilty of going right for each trial.
"""
function P_goright(model::choiceDDM)
    
    @unpack θ, data, dx, cross, initpt_mod = model
    @unpack θz, θhist = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    @unpack dt = data[1].click_data

    i_0 = compute_history(θhist, data, B)
    M,xc,n = initialize_latent_model(σ2_i, B, λ, σ2_a, dx, dt)

    pmap((data, i_0) -> likelihood!(θ, M, dx, xc, data, i_0, n, cross, initpt_mod), map(x-> choicedata(x.click_data, true), data), i_0)

end



"""
    likelihood!(θ, P, M, dx, xc, data, n, cross)

Given parameters θ and data (inputs and choices) computes the LL for one trial
"""
function likelihood!(θ::θchoice,
        M::Array{TT,2}, dx::Float64,
        xc::Vector{TT}, data::choicedata, i_0::TT,
        n::Int, cross::Bool, initpt_mod::Bool) where {TT <: Real}

    @unpack θz, bias, θlapse = θ
    @unpack click_data, choice = data

    initpt_mod ? a_0 = i_0 : a_0 = i_0*0.  
    
    P = P0(θz.σ2_i, a_0, n, dx, xc, click_data.dt)
    P = P_single_trial!(θz,P,M,dx,xc,click_data,n,cross)

    rlapse = get_rightlapse_prob(θlapse, i_0)
    @unpack lapse_prob = θlapse
    choice ? lapse_lik = rlapse : lapse_lik = (1-rlapse)
    sum(choice_likelihood!(bias,xc,P,choice,n,dx)) * (1 - lapse_prob) + (lapse_prob * lapse_lik)

end


"""
    P_single_trial!(θz, P, M, dx, xc, click_data, n)

Given parameters θz progagates P for one trial
"""
function P_single_trial!(θz,
        P::Vector{TT}, M::Array{TT,2}, dx::Float64,
        xc::Vector{TT}, click_data,
        n::Int, cross::Bool;
        keepP::Bool=false) where {TT <: Real}

    @unpack λ,σ2_a,σ2_s,ϕ,τ_ϕ = θz
    @unpack binned_clicks, clicks, dt = click_data
    @unpack nT, nL, nR = binned_clicks
    @unpack L, R = clicks

    #adapt magnitude of the click inputs
    La, Ra = adapt_clicks(ϕ,τ_ϕ,L,R; cross=cross)

    #empty transition matrix for time bins with clicks
    F = zeros(TT,n,n)
    
    if keepP
        PS = Vector{Vector{Float64}}(undef, nT)
    end

    @inbounds for t = 1:nT

        #maybe only pass one L,R,nT?
        P,F = latent_one_step!(P,F,λ,σ2_a,σ2_s,t,nL,nR,La,Ra,M,dx,xc,n,dt)
        
        if keepP
            PS[t] = P
        end


    end

    if keepP
        return PS
    else
        return P
    end

end


"""
    choice_likelihood!(bias, xc, P, pokedR, n, dx)

Preserves mass in the distribution P on the side consistent with the choice pokedR relative to the point bias. Deals gracefully in situations where the bias equals a bin center. However, if the bias grows larger than the bound, the LL becomes very large and the gradient is zero. However, it's general convexity of the -LL surface w.r.t this parameter should generally preclude it from approaches these regions.

### Examples

```jldoctest
julia> n, dt = 13, 1e-2;

julia> bias = 0.51;

julia> σ2_i, B, λ, σ2_a = 1., 2., 0., 10.; # the bound height of 2 is intentionally low, so P is not too long

julia> P, M, xc, dx = pulse_input_DDM.initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt);

julia> pokedR = true;

julia> round.(pulse_input_DDM.choice_likelihood!(bias, xc, P, pokedR, n, dx), digits=2)
13-element Array{Float64,1}:
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.04
 0.09
 0.08
 0.05
 0.03
 0.02
```
"""
function choice_likelihood!(bias::TT, xc::Vector{TT}, P::Vector{VV},
                 pokedR::Bool, n::Int, dx::Float64) where {TT,VV <: Any}

    lp = searchsortedlast(xc,bias)
    hp = lp + 1

    if ((hp==n+1) & (pokedR==true))
        P[1:lp-1] .= zero(TT)
        P[lp] = eps()

    elseif((lp==0) & (pokedR==false))
        P[hp+1:end] .= zero(TT)
        P[hp] = eps()

    elseif ((hp==n+1) & (pokedR==false)) || ((lp==0) & (pokedR==true))
        P .= one(TT)

    else

        dh, dl = xc[hp] - bias, bias - xc[lp]
        dd = dh + dl

        if pokedR
            P[1:lp-1] .= zero(TT)
            P[hp] = P[hp] * (1/2 + dh/dd/2)
            P[lp] = P[lp] * (dh/dd/2)
        else
            P[hp+1:end] .= zero(TT)
            P[hp] = P[hp] * (dl/dd/2)
            P[lp] = P[lp] * (1/2 + dl/dd/2)
        end

    end

    return P

end


"""
    get_rightlapse_prob(θlapse, i_0)

computes the probability of a rightward lapse based on trial history and general bias

"""
function get_rightlapse_prob(θlapse::θlapse, i_0)
    @unpack lapse_bias, lapse_modbeta = θlapse
    rbias =  1. ./(1. .+ exp.(-lapse_modbeta.*i_0 + lapse_bias))
    return rbias
end


"""
    compute_history(θhist, data, B)

computes the bias due to trial history on initial point/lapse probability
"""
function compute_history(θhist::θtrialhist, data, B::TT) where TT <: Any

    # this can be computed just once instead of at every iteration
    # and passed along in data (ask Brian!)
    choices = map(data -> data.choice, data)
    sessbnd = map(data -> data.click_data.sessbnd, data)
   
  #  correct = map(data -> data.click_data.clicks.gamma > 0, data)
    correct = map(data -> Δclicks(data.click_data)>0, data)
    hits = choices .== correct

    i_0 = Array{TT}(undef, length(correct))
    lim = 1
    for i = 1:length(correct)
        if sessbnd[i] == true
            lim, i_0[i] = i, 0.
        else
            k = compute_history(i, θhist, choices, hits, lim)     
            abs(k) > B ? i_0[i] = k * sign(B) : i_0[i] = k
        end
    end

    return i_0

end




"""
    choice_null(choices)

"""
choice_null(choices) = sum(choices .== true)*log(sum(choices .== true)/length(choices)) +
    sum(choices .== false)*log(sum(choices .== false)/length(choices))


# """
#     bounded_mass(θ, data, n)
# """
# function bounded_mass(θ::θchoice, data, dx::Float64, cross::Bool, initpt_mod::Bool)

#     @unpack θz, θhist = θ
#     @unpack σ2_i, B, λ, σ2_a = θz
#     @unpack dt = data[1].click_data

#     i_0 = compute_history(θhist, data, B)
#     initpt_mod ? a_0 = i_0 : a_0 = 0. * i_0
#     P,M,xc,dx = initialize_latent_model(σ2_i, a_0, B, λ, σ2_a, n, dt)

#     pmap(data -> bounded_mass!(θ, P, M, dx, xc, data, n, cross), data)

# end


# """
#     bounded_mass!(θ, P, M, dx, xc, data, n)
# """
# function bounded_mass!(θ::θchoice,
#         P::Vector{TT}, M::Array{TT,2}, dx::UU,
#         xc::Vector{TT}, data::choicedata,
#         n::Int, cross::Bool) where {TT,UU <: Real}

#     @unpack θz, bias = θ
#     @unpack click_data, choice = data

#     P = P_single_trial!(θz,P,M,dx,xc,click_data,n,cross)
#     choice ? P[n] : P[1]

# end


#=
Backward pass, for one day when I might need to compute the posterior again.

@inbounds for t = 1:T

    P,F = latent_one_step!(P,F,pz,t,hereL,hereR,La,Ra,M,dx,xc,n,dt)
    (t == T) && (P .*=  Pd)
    c[t] = sum(P)
    P /= c[t]
    comp_posterior ? post[:,t] = P : nothing

end

P = ones(Float64,n); #initialze backward pass with all 1's
post[:,T] .*= P;

@inbounds for t = T-1:-1:1

    (t + 1 == T) && (P .*=  Pd)
    P,F = latent_one_step!(P,F,pz,t+1,hereL,hereR,La,Ra,M,dx,xc,n,dt;backwards=true)
    P /= c[t+1]
    post[:,t] .*= P

end
=#
