"""
    P_goright(model)

Given an instance of `choiceDDM` computes the probabilty of going right for each trial.
"""
function P_goright(model::neural_choiceDDM, data)
    
    @unpack θ, n, cross = model
    @unpack θz, θy, bias, lapse = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    @unpack dt = data[1][1].input_data
       
    P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt)
    
    map((data, θy) -> pmap(data -> 
            P_goright(θ,θy,data,P,M,xc,dx,n,cross), data), data, θy)

end


"""
"""
function P_goright(θ, θy, data::neuraldata,
        P::Vector{T1}, M::Array{T1,2},
        xc::Vector{T1}, dx::T3, n, cross) where {T1,T3 <: Real}
    
    @unpack choice = data
    @unpack θz, bias, lapse = θ
    
    P = likelihood(θz, θy, data, P, M, xc, dx, n, cross)[2]
    sum(choice_likelihood!(bias,xc,P,true,n,dx)) * (1 - lapse) + lapse/2
    
end