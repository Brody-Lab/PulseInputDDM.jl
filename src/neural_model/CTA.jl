"""
"""
function optimize(data; filt_len::Int=50, Nfuns::Int=25, dt::Float64=1e-2)
    
    L,R = map(x-> binLR(x.input_data.binned_clicks, x.input_data.clicks, dt), data)
    L = getindex.(output, 1)
    R = getindex.(output, 2)
    L = makeX(L; filt_len=filt_len)
    R = makeX(R; filt_len=filt_len)
    
    spikes = map(n-> vcat(map(x-> x.spikes[n], data)...), 1:data[1].ncells)
    
    x = range(-dt*(filt_len-1), stop=0, length=filt_len)
    rbf = UniformRBFE(x, Nfuns, normalize=true) 
    
    m = BFGS(alphaguess = InitialStatic(alpha=1.0), linesearch = BackTracking())
    options = Optim.Options(g_tol=1e-4, x_tol=1e-6, f_tol=1e-9,
        iterations= 1_000, allow_f_increases=true, 
        show_trace = false, allow_outer_f_increases=true)
    
    results = Vector(undef,data[1].ncells)
    
    for n = 1:data[1].ncells
    
        function ℓℓ((w)
            wL, wR = rbf(x) * w[1:Nfuns], rbf(x) * w[Nfuns+1:end]
            -1*sum(loglikelihood.(Ref(wL), Ref(wR), L, R, spikes[n]; dt=dt))
        end

        w0 = 0.01 * randn(Nfuns*2);

        obj = OnceDifferentiable(ℓℓ, w0; autodiff=:forward)
        output = Optim.optimize(obj, ℓℓ, m, options)      
        results[n] = Optim.minimizer(output)
            
    end

    return results, x, rbf

end 
    
loglikelihood(wL, wR, L, R, spikes; dt::Float64=1e-2) = logpdf(Poisson(h(L, wL, R, wR)*dt), spikes)  
h(L, wL, R, wR) = exp(sum(skipmissing(wL .* L)) + sum(skipmissing(wR .* R)))
      
function makeX(X; filt_len::Int=10)
    
    vcat(map(X -> map(i-> vcat(missings(Int, max(0, filt_len - i)), X[max(1,i-filt_len+1):i]), 
                1:length(X)), X)...)
        
end
