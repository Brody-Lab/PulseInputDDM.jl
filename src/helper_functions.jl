
nanmean(x) = mean(filter(!isnan,x))
nanmean(x,y) = mapslices(nanmean,x,dims=y)

nanstderr(x) = std(filter(!isnan,x))/sqrt(length(filter(!isnan,x)))
nanstderr(x,y) = mapslices(nanstderr,x,dims=y)


"""
"""
function diffLR(click_data)
    
    @unpack binned_clicks, clicks, dt = click_data
    L,R = binLR(binned_clicks, clicks, dt)
    cumdiff = cumsum(-L + R)

    return cumdiff[end]
end


"""
"""
function binLR(binned_clicks, clicks, dt)

    @unpack L, R = clicks
    @unpack nT = binned_clicks

    t = 0:dt:nT*dt;
    L = fit(Histogram,L,t,closed=:left)
    R = fit(Histogram,R,t,closed=:left)
    L = L.weights
    R = R.weights

    return L,R
end



"""
get_param_names(θ)
given a θ <:DDMθ, returns an array with all the param names
"""
function get_param_names(θ::DDMθ)
    params = vcat(collect(fieldnames(typeof(θ.base_θz))), 
            collect(fieldnames(typeof(θ.ndtime_θz))),
            collect(fieldnames(typeof(θ.hist_θz))))
end


"""
CIs(H)
Given a Hessian matrix `H`, compute the 2 std confidence intervals based on the Laplace approximation.
If `H` is not positive definite (which it should be, but might not be due numerical round off, etc.) compute
a close approximation to it by adding a correction term. The magnitude of this correction is reported.
"""
function CIs(H::Array{Float64,2}) where T <: DDM

    HPSD = Matrix(cholesky(Positive, H, Val{false}))

    if !isapprox(HPSD,H)
        norm_ϵ = norm(HPSD - H)/norm(H)
        @warn "Hessian is not positive definite. Approximated by closest PSD matrix.
            ||ϵ||/||H|| is $norm_ϵ"
    end

    CI = 2*sqrt.(diag(inv(HPSD)))

    return CI, HPSD

end


"""
    stack(x,c)
Combine two vector into one. The first vector is variables for optimization, the second are constants.
"""
function stack(x::Vector{TT}, c::Vector{Float64}, fit::Union{BitArray{1},Vector{Bool}}) where TT

    v = Vector{TT}(undef,length(fit))
    v[fit] = x
    v[.!fit] = c

    return v

end


"""
    unstack(v)
Break one vector into two. The first vector is variables for optimization, the second are constants.
"""
function unstack(v::Vector{TT}, fit::Union{BitArray{1},Vector{Bool}}) where TT

    x,c = v[fit], v[.!fit]

end


"""
reconstruct_model(x, modeltype)
given a vector of params and modeltype, reconstructs θ
"""
function reconstruct_model(x::Vector{T1}, modeltype) where {T1 <: Real}
    if modeltype in keys(modeldict)
        return θ = Flatten.reconstruct(modeldict[modeltype](), x) 
    else
        error("Unknown model identifier $modeltype")
    end
 end
