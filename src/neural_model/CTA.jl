"""
"""
function optimize(data; filt_len::Int=50, Nfuns::Int=2, dt::Float64=1e-2)
    
    output = map(x-> binLR(x.input_data.binned_clicks, x.input_data.clicks, dt), data)
    L = getindex.(output, 1)
    R = getindex.(output, 2)
    nT = length.(L)
    xT = range(1, stop=maximum(nT), length=maximum(nT))
    ntrials = length(nT)
    
    m = BFGS(alphaguess = InitialStatic(alpha=1.0), linesearch = BackTracking())
    options = Optim.Options(g_tol=1e-4, x_tol=1e-6, f_tol=1e-9,
        iterations= 1_000, allow_f_increases=true, 
        show_trace = false, allow_outer_f_increases=true)
    
    LX = map(L -> map(i-> vcat(missings(Int, max(0, filt_len - i)), L[max(1,i-filt_len+1):i]), 1:length(L)), L)
    RX = map(R -> map(i-> vcat(missings(Int, max(0, filt_len - i)), R[max(1,i-filt_len+1):i]), 1:length(R)), R)
    spikes = map(x-> x.spikes[1], data)
    
    function ll(w, trials; α1::Float64=5e0)
    
        wL = w[1:filt_len]
        wR = w[filt_len+1:2*filt_len]
        #w0 = w[2*filt_len+1:2*filt_len + maximum(nT)]
        w0 = Polynomials.Poly(w[2*filt_len+1:2*filt_len+Nfuns])(xT)

        w0X = map(t-> w0[1:t], nT[trials])

        -(sum(loglikelihood.(Ref(wL), Ref(wR), vcat(LX[trials]...), vcat(RX[trials]...), 
                    vcat(w0X...), vcat(spikes[trials]...); dt=dt)) - 
            α1 * sum(map(x-> sum(diff(x).^2), [wL, wR])) -
            α1 * sum(map(x-> sum(diff(diff(x)).^2), [wL, wR])))

    end
        
    function optimize(α1, trials)
            
        ℓℓ(w) = ll(w, trials; α1=α1)
        winit = 0.01 * randn(2*filt_len + Nfuns)    
        obj = OnceDifferentiable(ℓℓ, winit; autodiff=:forward)
        output = Optim.optimize(obj, winit, m, options)
        Optim.minimizer(output)
            
    end
        
    function batch(i)
        
        train = sample(Random.seed!(i), 1:ntrials, ceil(Int, 0.9 * ntrials), replace=false)
        test = setdiff(1:ntrials, train)
        #α1s = 10. .^((-6:7) .+ (-1 .+ 2 * rand(14)))
        α1s = 10. .^(-6:7)
                  
        output = pmap(α1-> optimize(α1, train), α1s)   
        testLL = pmap(w-> -(ll(w, test; α1=0.)), output)
        
        return α1s, output, testLL
        
    end
    
    pmap(i-> batch(i), 1:10)  
            
end 
    
loglikelihood(wL, wR, L, R, w0, spikes; dt::Float64=1e-2) = logpdf(Poisson(h(L, wL, R, wR, w0)*dt), spikes)
h(L, wL, R, wR, w0) = softplus(sum(skipmissing(wL .* L)) + sum(skipmissing(wR .* R)) + w0)
