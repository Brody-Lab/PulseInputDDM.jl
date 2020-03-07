"""
"""
@with_kw struct filtinputs{T1,T2,T3,T4}
    clicks::T1
    binned_clicks::T2
    LX::T3
    RX::T4
    dt::Float64
    centered::Bool
end


"""
"""
@with_kw struct filtdata <: DDMdata
    input_data::filtinputs
    spikes::Vector{Vector{Int}}
    ncells::Int
end



"""
"""
@with_kw struct sigmoid_filtoptions <: neural_options
    ncells::Vector{Int}
    nparams::Int = 4
    npolys::Int = 4
    filt_len::Int = 50
    f::String = "Sigmoid"
    fit::Vector{Bool} = vcat(trues(2*filt_len + sum(ncells)*npolys + sum(ncells)*nparams))
    lb::Vector{Float64} = vcat(-Inf*ones(2*filt_len),
        repeat(-Inf * ones(npolys), sum(ncells)),
        repeat([-100.,0.,-10.,-10.], sum(ncells)))
    ub::Vector{Float64} = vcat(Inf*ones(2*filt_len),
        repeat(Inf * ones(npolys), sum(ncells)),
        repeat([100.,100.,10.,10.], sum(ncells)))
    x0::Vector{Float64} = vcat(0.01 * randn(2*filt_len),
        repeat(zeros(npolys), sum(ncells)),
        repeat([10.,10.,1.,0.], sum(ncells)))
end


"""
"""
@with_kw struct filtoptions <: neural_options
    ncells::Vector{Int}
    nparams::Int = 3
    npolys::Int = 4
    filt_len::Int = 50
    f::String = "Softplus"
    fit::Vector{Bool} = vcat(trues(2*filt_len + sum(ncells)*npolys + sum(ncells)*nparams))
    lb::Vector{Float64} = vcat(-Inf*ones(2*filt_len),
        repeat(-Inf * ones(npolys), sum(ncells)),
        repeat([eps(), -Inf, -Inf], sum(ncells)))
    ub::Vector{Float64} = vcat(Inf*ones(2*filt_len),
        repeat(Inf * ones(npolys), sum(ncells)),
        repeat([Inf, Inf, Inf], sum(ncells)))
    x0::Vector{Float64} = vcat(0.01 * randn(2*filt_len),
        repeat(zeros(npolys), sum(ncells)),
        repeat([10.,1.,0.], sum(ncells)))
end


"""
"""
@flattenable @with_kw struct θfilt{T1, T2, T3} <: DDMθ
    θτ::T1 = θτ() | true
    θμ::T2 | true
    θy::T3 | true
    ncells::Vector{Int} | false
    nparams::Int
    f::String
    npolys::Int
end


"""
"""
@with_kw struct θτ{T1,T2}
    filt_len::Int = 50
    wL::T1 = 0.01 * randn(filt_len)
    wR::T2 = 0.01 * randn(filt_len)
end



"""
"""
function train_and_test(data, options::Union{sigmoid_filtoptions, filtoptions}; seed::Int=1, α1s = 10. .^(-6:7))
    
    @unpack filt_len = options

    ntrials = length(data)
    train = sample(Random.seed!(seed), 1:ntrials, ceil(Int, 0.9 * ntrials), replace=false)
    test = setdiff(1:ntrials, train)
    
    model = pmap(α1-> optimize([data[train]], options; α1=α1, show_trace=false), α1s)   
    
    filt_data = make_filt_data.(data, Ref(filt_len));
    testLL = pmap(model-> loglikelihood(model.θ, [filt_data[test]]), model)

    return α1s, model, testLL
    
end


"""
"""
function make_filt_data(data, filt_len)
    
    @unpack input_data, spikes, ncells = data
    @unpack binned_clicks, clicks, dt, centered = input_data

    L,R = binLR(binned_clicks, clicks, dt)
    LX = map(i-> vcat(missings(Int, max(0, filt_len - i)), L[max(1,i-filt_len+1):i]), 1:length(L))
    RX = map(i-> vcat(missings(Int, max(0, filt_len - i)), R[max(1,i-filt_len+1):i]), 1:length(R));

    filtdata(filtinputs(clicks, binned_clicks, LX, RX, dt, centered), spikes, ncells)
    
end



"""
"""
function optimize(data, options::Union{sigmoid_filtoptions, filtoptions};
        x_tol::Float64=1e-10, f_tol::Float64=1e-9, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        outer_iterations::Int=Int(1e1), α1::Float64=10. .^0)

    @unpack fit, lb, ub, x0, ncells, f, nparams, npolys, filt_len = options
    
    filt_data = map(data-> make_filt_data.(data, Ref(filt_len)), data)

    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)
    #ℓℓ(x) = -(loglikelihood(stack(x,c,fit), filt_data, ncells, nparams, npolys, filt_len, f) - 
    #        α1 * sum(diff(stack(x,c,fit)[1:2*filt_len]).^2) -
    #        α1 * sum(diff(diff(stack(x,c,fit)[1:2*filt_len])).^2))
    
    #ℓℓ(x) = -(loglikelihood(stack(x,c,fit), filt_data, ncells, nparams, npolys, filt_len, f) - 
    #    α1 * sum(diff(stack(x,c,fit)[1:2*filt_len]).^2) - 
    #    1e0 * (sum(stack(x,c,fit)[1:2*filt_len]) - 1.))
    
    ℓℓ(x) = -(loglikelihood(stack(x,c,fit), filt_data, ncells, nparams, npolys, filt_len, f) - 
        α1 * sum(diff(stack(x,c,fit)[1:2*filt_len]).^2))

    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    θ = unflatten(x, ncells, nparams, f, npolys, filt_len)
    model = neuralDDM(θ, data)
    converged = Optim.converged(output)

    return model

end


"""
    loglikelihood(x, data, ncells)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input. Used
in optimization, Hessian and gradient computation.
"""
function loglikelihood(x::Vector{T1}, data, ncells, nparams, npolys, filt_len::Int, f::String) where {T1 <: Real}

    θ = unflatten(x, ncells, nparams, f, npolys, filt_len)
    loglikelihood(θ, data)

end


"""
"""
function unflatten(x::Vector{T}, ncells::Vector{Int}, nparams::Int, f::String, npolys::Int, filt_len::Int) where {T <: Real}
    
    dims2 = vcat(0,cumsum(ncells))

    blah = Tuple.(collect(partition(x[2*filt_len + 
                npolys*sum(ncells) + 1:2*filt_len + npolys*sum(ncells) + nparams*sum(ncells)], nparams)))
    
    if f == "Sigmoid"
        blah2 = map(x-> Sigmoid(x...), blah)
    elseif f == "Softplus"
        blah2 = map(x-> Softplus(x...), blah)
    end
    
    θy = map(idx-> blah2[idx], [dims2[i]+1:dims2[i+1] for i in 1:length(dims2)-1]) 
    
    blah = Tuple.(collect(partition(x[2*filt_len+1:2*filt_len+npolys*sum(ncells)], npolys)))
    
    blah2 = map(x-> Poly(collect(x)), blah)
    
    θμ = map(idx-> blah2[idx], [dims2[i]+1:dims2[i+1] for i in 1:length(dims2)-1])
    
    θfilt(θτ(filt_len, x[1:filt_len], x[filt_len+1:2*filt_len]), θμ, θy, ncells, nparams, f, npolys)

end


"""
"""
function loglikelihood(θ::θfilt, data)

    @unpack θτ, θμ, θy = θ

    sum(map((θy, θμ, data) -> sum(loglikelihood.(Ref(θτ), Ref(θμ), Ref(θy), data)), θy, θμ, data))

end


"""
"""
function loglikelihood(θτ::θτ, θμ, θy, data::filtdata)

    @unpack spikes, input_data = data
    @unpack dt = input_data
    λ, = loglikelihood(θτ,θμ,θy,input_data)
    sum(logpdf.(Poisson.(vcat(λ...)*dt), vcat(spikes...)))

end



"""
"""
function loglikelihood(θτ::θτ, θμ, θy, input_data::filtinputs)

    @unpack binned_clicks, dt = input_data
    @unpack nT = binned_clicks

    a = rand(θτ,input_data)
    λ = map((θy,θμ)-> θy(a, θμ(1:nT)), θy, θμ)

    return λ, a

end


"""
"""
function rand(θτ::θτ, inputs)

    @unpack wL, wR, filt_len = θτ
    @unpack LX, RX = inputs
    #@unpack clicks, binned_clicks, dt = inputs
    #@unpack nT = binned_clicks
        
    #L,R = binLR(binned_clicks, clicks, dt)
    
    #LX = map(i-> vcat(missings(Int, max(0, filt_len - i)), L[max(1,i-filt_len+1):i]), 1:length(L))
    #RX = map(i-> vcat(missings(Int, max(0, filt_len - i)), R[max(1,i-filt_len+1):i]), 1:length(R)) 
    
    #a = map((L,R)-> sum(skipmissing(wL .* L)) + sum(skipmissing(wR .* R)), LX, RX)   
    afilt.(Ref(wL), LX, Ref(wR), RX)      
    
end


"""
"""
afilt(wL, L, wR, R) = sum(skipmissing(wL .* L)) + sum(skipmissing(wR .* R))    


"""
    Sample rates from latent model with multiple rngs, to average over
"""
function synthetic_λ(θ::θfilt, data; nconds::Int=2)

    @unpack θτ,θμ,θy,ncells = θ

    μ_λ = rand.(Ref(θτ), θμ, θy, data)
    
    μ_c_λ = cond_mean.(μ_λ, data, ncells; nconds=nconds)
    
    return μ_λ, μ_c_λ

end


"""
    Sample all trials over one session
"""
function rand(θτ::θτ, θμ, θy, data)
    
    pmap(data -> loglikelihood(θτ,θμ,θy,data.input_data)[1], data)

end