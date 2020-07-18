
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
"""
function transform_log_space(θ::DDMθ, teps::Float64; nsamples::Int = 80000) 

    σ2_s = θ.base_θz.σ2_s
    if θ.lpost_space == 1
        d = fit(Normal, -teps .+ (2*teps)*cdf.(Normal(0, σ2_s), rand(Normal(1, sqrt(σ2_s)),nsamples)))
        return (d.σ)^2, d.μ
    else
        return σ2_s, "undef" 
    end
 
end

"""
"""
function evidence_no_noise(gamma; dteps::Float64 = 1e-50)

    g = sort(unique(gamma))
    gprob = round.([count(x->x==i,gamma) for i in g]./length(gamma), digits = 1)
    
    R = 40*dteps
    rrate = R.*exp.(g)./(exp.(g) .+ 1);
    lrate = R .- rrate;
  
    g_nonoise = map((rrate, lrate, gprob)-> rrate*(1-lrate)*gprob, rrate, lrate, gprob)
    return log.(sum(g_nonoise[g.>0])./sum(g_nonoise[g.<0]))

end

"""
    constraint_penalty when fitting unconstrained exponential in log-posterior space

"""
function constraint_penalty(hist_θz::θz_LPSexp, ntrials::Int, meanll::Float64)

    @unpack h_α, h_β, h_C = hist_θz

    reg = InverseGamma(0.001, 0.1)
    ep = 0.0001344
    dum = 1. - h_C - h_β - h_α

    if dum < ep
        intercept = log(pdf(reg, ep))
        slope = -(log(pdf(reg,ep)) - log(pdf(reg, 2*ep)))/ep
        return slope*(dum-ep) + intercept, false
    else
        return log(pdf(reg,dum)), true
    end
end


"""
    constraint_penalty for all other models

"""
function constraint_penalty(hist_θz, ntrials, meanll)
    return 0, true
end


"""
get_param_names(θ)
given a θ <:DDMθ, returns an array with all the param names
"""
function get_param_names(θ::DDMθ)
    params = vcat(collect(fieldnames(typeof(θ.base_θz))), 
            collect(fieldnames(typeof(θ.ndtime_θz))),
            collect(fieldnames(typeof(θ.hist_θz))),
            collect(fieldnames(typeof(θ)))[4:end])  #lpostspace
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


# """
# reconstruct_model(x, modeltype)
# given a vector of params and modeltype, reconstructs θ
# """
# function reconstruct_model(x::Vector{T1}, modeltype::String) where {T1 <: Any}
#     if modeltype in keys(modeldict)
#         return θ = Flatten.reconstruct(modeldict[modeltype](), x) 
#     else
#         error("Unknown model identifier $modeltype")
#     end
#  end
