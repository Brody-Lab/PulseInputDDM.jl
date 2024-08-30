"""
"""
mutable struct θμ_RBF{T1} <: DDMθ
    θμ::T1
    ncells::Vector{Int}
    nRBFs::Int
end


function optimize(data, ncells::Vector{Int}; nRBFs::Int = 6, 
        x_tol::Float64=1e-10, f_tol::Float64=1e-6, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        outer_iterations::Int=Int(1e1))

    fit = trues(sum(ncells)*nRBFs)
    lb  = repeat(zeros(nRBFs), sum(ncells))
    ub = repeat(Inf * ones(nRBFs), sum(ncells))
    x0 = repeat([1. for i in 0:(nRBFs-1)], sum(ncells))
    
    θ = θμ_RBF(x0, ncells, nRBFs)
    ℓℓ(x) = -loglikelihood(x, data, θ)

    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations)

    x = Optim.minimizer(output)
    θ = θμ_RBF(x, ncells, nRBFs)

end


function loglikelihood(x::Vector{T1}, data::Vector{Vector{T2}}, θ::θμ_RBF) where {T1 <: Real, T2 <: neuraldata}
    
    @unpack ncells, nRBFs = θ
    θ = θμ_RBF(x, ncells, nRBFs)
    loglikelihood(θ, data)

end


"""
"""
function θμ_RBF(x::Vector{T}, ncells::Vector{Int}, nRBFs::Int) where {T <: Real}
    
    dims2 = vcat(0,cumsum(ncells))   
    blah = Tuple.(collect(partition(x, nRBFs)))    
    blah2 = map(x-> collect(x), blah)  
    θμ = map(idx-> blah2[idx], [dims2[i]+1:dims2[i+1] for i in 1:length(dims2)-1])   
    θμ_RBF(θμ, ncells, nRBFs)

end


"""
"""
function loglikelihood(θ::θμ_RBF, data::Vector{Vector{T1}}) where {T2 <: Real, T1 <: neuraldata}
    
    @unpack θμ, nRBFs = θ
    
    pad = data[1][1].input_data.pad   
    maxnT = maximum(vcat(map(data-> map(data-> data.input_data.binned_clicks.nT, data), data)...))   
    x = 1:maxnT+2*pad   
    rbf = UniformRBFE(x, nRBFs, normalize=true)     

    sum(map((θμ, data) -> sum(loglikelihood.(Ref(θμ), data, Ref(rbf))), θμ, data))

end


"""
"""
function loglikelihood(θμ::Vector{Vector{T2}}, 
        data::neuraldata, rbf) where {T2 <: Real}

    @unpack spikes, input_data = data
    @unpack binned_clicks, dt, pad = input_data
    @unpack nT = binned_clicks
    
    x = 1:nT+2*pad   
    #λ = map(θμ-> max.(0., rbf(x) * θμ), θμ)     
    #λ = map(θμ-> softplus.(rbf(x) * θμ), θμ)
    λ = map(θμ-> rbf(x) * θμ, θμ)     

    sum(logpdf.(Poisson.(vcat(λ...)*dt), vcat(spikes...)))

end