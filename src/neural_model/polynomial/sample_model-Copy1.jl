#=
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
=#

"""
    Sample rates from latent model with multiple rngs, to average over
"""
function synthetic_λ(θ::θneural, data; num_samples::Int=100, nconds::Int=2)

    @unpack θz,θy,ncells = θ

    λ = map(rng-> rand.(Ref(θz), θy, data, Ref(rng)), 1:num_samples)
    μ_λ = mean(λ)
    
    μ_c_λ = cond_mean.(μ_λ, data, ncells; nconds=nconds)
    
    return μ_λ, μ_c_λ

end


"""
    Sample rates from latent model with multiple rngs, to average over
"""
function synthetic_λ(θ::θneural_poly, data; num_samples::Int=100, nconds::Int=2)

    @unpack θz,θμ,θy,ncells = θ

    λ = map(rng-> rand.(Ref(θz), θμ, θy, data, Ref(rng)), 1:num_samples)
    μ_λ = mean(λ)
    
    μ_c_λ = cond_mean.(μ_λ, data, ncells; nconds=nconds)
    
    return μ_λ, μ_c_λ

end


"""
    Sample all trials over one session
"""
function rand(θz::θz, θy::θy, data, rng)
    
    ntrials = length(data)
    rng = sample(Random.seed!(rng), 1:ntrials, ntrials; replace=false)

    pmap((data,rng) -> rand(θz,θy,data.input_data; rng=rng)[1], data, rng)

end


"""
    Sample all trials over one session
"""
function rand(θz::θz, θμ, θy::θy, data, rng)
    
    ntrials = length(data)
    rng = sample(Random.seed!(rng), 1:ntrials, ntrials; replace=false)

    pmap((data,rng) -> rand(θz,θμ,θy,data.input_data; rng=rng)[1], data, rng)

end


"""
"""
function cond_mean(μ_λ, data, ncells; nconds=2)
        
    nT = map(x-> x.input_data.binned_clicks.nT, data)
    ΔLRT = last.(diffLR.(data))
    conds = encode(LinearDiscretizer(binedges(DiscretizeUniformWidth(nconds), ΔLRT)), ΔLRT)

    map(n-> map(c-> [mean([μ_λ[conds .== c][k][n][t]
        for k in findall(nT[conds .== c] .>= t)])
        for t in 1:(maximum(nT[conds .== c]))],
                1:nconds), 1:ncells)

end


"""
"""
function synthetic_data(θ::θneural,
        ntrials::Vector{Int}; centered::Bool=true,
        dt::Float64=1e-2, rng::Int=1, dt_synthetic::Float64=1e-4, pad::Int=10)

    nsess = length(ntrials)
    rng = sample(Random.seed!(rng), 1:nsess, nsess; replace=false)

    @unpack θz,θμ,θy,ncells = θ

    output = rand.(Ref(θz), θμ, θy, ntrials, ncells, rng)

    spikes = getindex.(output, 1)
    clicks = getindex.(output, 2)

    output = bin_clicks_spikes_λ0.(spikes, clicks;
        centered=centered, dt=dt, dt_synthetic=dt_synthetic, synthetic=true)

    spikes = getindex.(output, 1)
    binned_clicks = getindex.(output, 2)

    input_data = neuralinputs.(clicks, binned_clicks, dt, centered)
    
    padded = map(spikes-> map(spikes-> map(SCn-> vcat(rand.(Poisson.((sum(SCn[1:10])/(10*dt))*ones(pad)*dt)), 
                    SCn, rand.(Poisson.((sum(SCn[end-9:end])/(10*dt))*ones(pad)*dt))), spikes), spikes), spikes)
    
    μ_rnt = map(padded-> filtered_rate.(padded, dt), padded)
    
    nT = map(x-> map(x-> x.nT, x), binned_clicks)
    
    μ_t = map((μ_rnt, ncells, nT)-> map(n-> [max(0., mean([μ_rnt[i][n][t]
        for i in findall(nT .>= t)]))
        for t in 1:(maximum(nT))], 1:ncells), μ_rnt, ncells, nT)

    neuraldata.(input_data, spikes, ncells), μ_rnt, μ_t

end


"""
"""
function rand(θz::θz, θy::θy, ntrials, ncells, rng; centered::Bool=false, dt::Float64=1e-4, pos_ramp::Bool=false)

    clicks = synthetic_clicks.(ntrials, rng)
    binned_clicks = bin_clicks.(clicks, centered=centered, dt=dt)
    λ0 = synthetic_λ0.(clicks, ncells; dt=dt, pos_ramp=pos_ramp)
    input_data = neuralinputs.(clicks, binned_clicks, λ0, dt, centered)

    rng = sample(Random.seed!(rng), 1:ntrials, ntrials; replace=false)

    spikes = pmap((input_data,rng) -> rand(θz,θy,input_data; rng=rng)[3], input_data, rng)

    return spikes, λ0, clicks

end



"""
"""
function rand(θz::θz, θμ, θy::θy, ntrials, ncells, rng; centered::Bool=false, dt::Float64=1e-4)

    clicks = synthetic_clicks.(ntrials, rng)
    binned_clicks = bin_clicks.(clicks, centered=centered, dt=dt)
    input_data = neuralinputs.(clicks, binned_clicks, dt, centered)

    rng = sample(Random.seed!(rng), 1:ntrials, ntrials; replace=false)

    spikes = pmap((input_data,rng) -> rand(θz,θμ,θy,input_data; rng=rng)[3], input_data, rng)

    return spikes, clicks

end


"""
"""
function rand(θz::θz, θy::θy, input_data::neuralinputs; rng::Int=1)

    @unpack λ0, dt = input_data

    Random.seed!(rng)
    a = rand(θz,input_data)
    λ = map((θy,λ0)-> θy(a, λ0), θy, λ0)
    spikes = map(λ-> rand.(Poisson.(λ*dt)), λ)

    return λ, a, spikes

end



"""
"""
function rand(θz::θz, θμ, θy::θy, input_data::neuralinputs; rng::Int=1)

    @unpack binned_clicks, dt = input_data
    @unpack nT = binned_clicks
    
    Random.seed!(rng)
    a = rand(θz,input_data)
    λ = map((θy,θμ)-> θy(a, θμ(1:nT)), θy, θμ)
    spikes = map(λ-> rand.(Poisson.(λ*dt)), λ)

    return λ, a, spikes

end
