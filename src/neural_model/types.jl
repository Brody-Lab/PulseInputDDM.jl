"""
"""
@with_kw struct neuralinputs{T1,T2}
    clicks::T1
    binned_clicks::T2
    λ0::Vector{Vector{Float64}}
    dt::Float64
    centered::Bool
    delay::Int
    pad::Int
end


"""
"""
neuralinputs(clicks, binned_clicks, λ0::Vector{Vector{Vector{Float64}}}, dt::Float64, centered::Bool, delay::Int, pad::Int) =
    neuralinputs.(clicks, binned_clicks, λ0, dt, centered, delay, pad)


"""
"""
@with_kw struct θneural{T1, T2} <: DDMθ
    θz::T1
    θy::T2
    f::Vector{Vector{String}}
end


"""
    neuralDDM

Fields:
- θ
- n
- cross
"""
@with_kw struct neuralDDM{T} <: DDM
    θ::T
    n::Int=53
    cross::Bool=false
end


"""
"""
@with_kw struct θneural_noiseless{T1, T2} <: DDMθ
    θz::T1
    θy::T2
    f::Vector{Vector{String}}
end


"""
"""
@with_kw struct noiseless_neuralDDM{T} <: DDM
    θ::T
end


"""
"""
@with_kw struct neural_options
    fit::Vector{Bool}
    ub::Vector{Float64}
    lb::Vector{Float64}
end



"""
"""
function neural_options(f)
    
    nparams, ncells = nθparams(f)
    fit = vcat(trues(dimz), trues.(nparams)...)
        
    lb = Vector(undef, sum(ncells))
    ub = Vector(undef, sum(ncells))
    
    for i in 1:sum(ncells)
        if vcat(f...)[i] == "Softplus"
            lb[i] = [-10]
            ub[i] = [10]
        elseif vcat(f...)[i] == "Sigmoid"
            lb[i] = [-100.,0.,-10.,-10.]
            ub[i] = [100.,100.,10.,10.]
        end
    end
    lb = vcat([0., 4.,  -5., 0.,   0.,  0.01, 0.005], vcat(lb...))
    ub = vcat([30., 30., 5., 100., 2.5, 1.2,  1.], vcat(ub...));

    neural_options(fit=fit, ub=ub, lb=lb)
    
end
   

"""
"""
function θneural(x::Vector{T}, f::Vector{Vector{String}}) where {T <: Real}
    
    nparams, ncells = nθparams(f)
    
    borg = vcat(dimz,dimz.+cumsum(nparams))
    blah = [x[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]
    
    blah = map((f,x) -> f(x...), getfield.(Ref(@__MODULE__), Symbol.(vcat(f...))), blah)
    
    borg = vcat(0,cumsum(ncells))
    θy = [blah[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]
    
    θneural(θz(x[1:dimz]...), θy, f)

end


"""
"""
@with_kw struct θy{T1}
    θ::T1
end


"""

    neuraldata

Module-defined class for keeping data organized for the `neuralDDM` model.

Fields:

- `input_data`: stuff related to the input of the accumaltor model, i.e. clicks, etc.
- `spikes`: the binned spikes
- `ncells`: numbers of cells on that trial (should be the same for every trial in a session)
- `choice`: choice on that trial

"""
@with_kw struct neuraldata <: DDMdata
    input_data::neuralinputs
    spikes::Vector{Vector{Int}}
    ncells::Int
    choice::Bool
end


"""
"""
neuraldata(input_data, spikes::Vector{Vector{Vector{Int}}}, ncells::Int, choice) =  neuraldata.(input_data,spikes,ncells,choice)


"""
"""
@with_kw struct Sigmoid{T1} <: DDMf
    a::T1=10.
    b::T1=10.
    c::T1=1.
    d::T1=0.
end


"""
"""
(θ::Sigmoid)(x::Vector{U}, λ0::Vector{T}) where {U,T <: Real} =
    (θ::Sigmoid).(x, λ0)


"""
"""
function (θ::Sigmoid)(x::U, λ0::T) where {U,T <: Real}

    @unpack a,b,c,d = θ

    y = c * x + d
    y = a + b * logistic!(y) + λ0
    y = softplus(y)

end


@with_kw struct Softplussign{T1} <: DDMf
    #a::T1 = 0
    c::T1 = 5.0*rand([-1,1])
end


"""
"""
function (θ::Softplussign)(x::Union{U,Vector{U}}, λ0::Union{T,Vector{T}}) where {U,T <: Real}

    #@unpack a,c = θ
    @unpack c = θ

    #y = a .+ softplus.(c*x .+ d) .+ λ0
     #y = softplus.(c*x .+ a .+ λ0)
     y = softplus.(c .* sign.(x) .+ softplusinv.(λ0))
    #y = max.(eps(), y .+ λ0)
    #y = softplus.(y .+ λ0)
end

"""
   Softplus(c)

``\\lambda(a) = \\ln(1 + \\exp(c * a))``
"""
@with_kw struct Softplus{T1} <: DDMf
    #a::T1 = 0
    c::T1 = 5.0*rand([-1,1])
end


"""
"""
function (θ::Softplus)(x::Union{U,Vector{U}}, λ0::Union{T,Vector{T}}) where {U,T <: Real}

    #@unpack a,c = θ
    @unpack c = θ

    #y = a .+ softplus.(c*x .+ d) .+ λ0
     #y = softplus.(c*x .+ a .+ λ0)
     y = softplus.(c*x .+ softplusinv.(λ0))
    #y = max.(eps(), y .+ λ0)
    #y = softplus.(y .+ λ0)
end

softplusinv(x) = log(expm1(x))

