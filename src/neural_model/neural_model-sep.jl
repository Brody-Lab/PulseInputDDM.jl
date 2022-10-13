"""
    sep_neural_loglikelihood(x, model)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input. Used
in optimization, Hessian and gradient computation.
"""
function sep_neural_loglikelihood(x::Vector{T}, model::neuralDDM) where {T <: Real}
    
    @unpack data,θ,n,cross = model
    @unpack f = θ 
    
    model = neuralDDM(θ=θneural(x, f), data=data, n=n, cross=cross)
    
    sep_neural_loglikelihood(model)

end


"""
    sep_neural_loglikelihood(model)

Given parameters θ and data (inputs and choices) computes the LL for all trials
"""
sep_neural_loglikelihood(model::neuralDDM) = sum(log.(vcat(vcat(sep_neural_likelihood(model)...)...)))


"""
    sep_neural_likelihood(model)

Arguments: `neuralDDM` instance

Returns: `array` of `array` of P(d, Y|θ)
"""
function sep_neural_likelihood(model::neuralDDM)
    
    @unpack data,θ,n,cross = model
    @unpack θz, θy = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    @unpack dt = data[1][1].input_data

    P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt)

    map((data, θy) -> pmap(data -> 
            sep_neural_likelihood(θ,θy,data,P,M,xc,dx,n,cross), data), data, θy)
    
end


"""
"""
function sep_neural_likelihood(θ, θy, data::neuraldata,
        P::Vector{T1}, M::Array{T1,2},
        xc::Vector{T1}, dx::T3, n, cross) where {T1,T3 <: Real}
    
    @unpack θz = θ
    @unpack spikes, input_data = data
    @unpack λ0 = input_data
    
    output = map((θy, spikes, λ0) -> likelihood(θz, [θy], [spikes], input_data, [λ0], P, M, xc, dx, n, cross), θy, spikes, λ0)
    c = getindex.(output, 1)    

    return vcat(c...)
     
end