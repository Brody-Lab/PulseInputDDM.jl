
nanmean(x) = mean(filter(!isnan,x))
nanmean(x,y) = mapslices(nanmean,x,dims=y)

nanstderr(x) = std(filter(!isnan,x))/sqrt(length(filter(!isnan,x)))
nanstderr(x,y) = mapslices(nanstderr,x,dims=y)

#should modify this to return the differnece only, and then allow filtering afterwards
function diffLR(nT,L,R,dt)
    
    L,R = binLR(nT,L,R,dt)   
    cumsum(-L + R)
    
end

function load_and_dprime(path::String, sessids, ratnames; 
        dt::Float64=1e-3, delay::Float64=0.)
    
    data = aggregate_spiking_data(path,sessids,ratnames)
    data = map(x->bin_clicks_spikes_and_λ0!(x; dt=dt,delay=delay), data)
    
    map(d-> map(n-> dprime(map(r-> data[d]["μ_rn"][r][n], 1:data[d]["ntrials"]), data[d]["pokedR"]), 
            1:data[d]["N"]),
                1:length(data))
        
end

function binLR(nT,L,R,dt)
    
    #compute the cumulative diff of clicks
    t = 0:dt:nT*dt;
    L = fit(Histogram,L,t,closed=:left)
    R = fit(Histogram,R,t,closed=:left)
    L = L.weights
    R = R.weights
    
    return L,R
    
end

function predict_choice_Y(pz, py, bias, data; dt::Float64=1e-2, n::Int=53, f_str::String="softplus",
        λ0::Vector{Vector{Vector{Float64}}}=Vector{Vector{Vector{Float64}}}()) 

    PS = pulse_input_DDM.PY_all_trials(pz, py, data; λ0=λ0, f_str=f_str)
    
    xc,dx,xe = pulse_input_DDM.bins(pz[2],n);
    nbinsL, Sfrac = pulse_input_DDM.bias_bin(bias,xe,dx,n)
    Pd = vcat(1. * ones(nbinsL), 1. * Sfrac + 0. * (1. - Sfrac), 0. * ones(n - (nbinsL + 1)))

    predicted_choice = map(x-> !Bool(round(sum(x[:, end] .* Pd))) , PS)
    
    per_corr = sum(predicted_choice .== data["pokedR"]) / length(data["pokedR"])
    
    return per_corr, predicted_choice
    
end

function predict_choice(pz, bias, data; n::Int=53)

    PS = P_all_trials(pz, data)
    
    xc,dx,xe = pulse_input_DDM.bins(pz[2],n);
    nbinsL, Sfrac = pulse_input_DDM.bias_bin(bias,xe,dx,n)
    Pd = vcat(1. * ones(nbinsL), 1. * Sfrac + 0. * (1. - Sfrac), 0. * ones(n - (nbinsL + 1)))

    predicted_choice = map(x-> !Bool(round(sum(x[:, end] .* Pd))) , PS)
    
    per_corr = sum(predicted_choice .== data["pokedR"]) / length(data["pokedR"])
    
    return per_corr, predicted_choice
    
end

function dprime(FR,choice)
    abs(mean(FR[choice .== false]) - mean(FR[choice .== true])) / 
        sqrt(0.5 * (var(FR[choice .== false])^2 + var(FR[choice .== true])^2))
end

#function padded_λ_array(nspikes,dt,filtSD)
    
#    nT = length.(nspikes)
#    λ = fill!(Array{Float64,2}(undef, length(nspikes), maximum(nT)),NaN)
    
#    kern = reflect(KernelFactors.gaussian(filtSD,8*filtSD+1));
#    for i = 1:length(nspikes)
#        λ[i,1:nT[i]] = (1/dt) * imfilter(nspikes[i], kern)
#    end
    
#    return λ
    
#end

function rate_mat_func_filt(nspikes,dt,filtSD)
    
    nT = map(length,nspikes)
    λ = fill!(Array{Float64,2}(undef,length(nspikes),maximum(nT)),NaN);
    
    for i = 1:length(nspikes)
        λ[i,1:nT[i]] = vec(FilterSpikes((1/dt)*nspikes[i],filtSD,pad="zeros"))
    end
    
    return λ
        
end

function FilterSpikes(x,SD;pad::String="zeros")
    
    #plot(conv(pdf.(Normal(mu1,sigma1),x1),pdf.(Normal(mu2,sigma2),x2)))
    #plot(filt(digitalfilter(PolynomialRatio(gaussian(10,1e-2),ones(10)),pdf.(Normal(mu1,sigma1),x1)))
    #plot(pdf.(Normal(mu1,sigma1),x1))
    #plot(filt(digitalfilter(Lowpass(0.8),FIRWindow(gaussian(100,0.5))),pdf.(Normal(mu1,sigma1),x1)))
    #plot(filtfilt(gaussian(100,0.02), pdf.(Normal(mu1,sigma1),x1) + pdf.(Normal(mu2,sigma2),x1)))
    #x = round.(-0.45+rand(1000))
    #plot((1/1e-3)*(1/100)*filt(rect(100),x));
    #plot(gaussian(100,0.5))
    #gaussian(10000,0.5)
    #plot((1/1e-3)*filt((1/sum(pdf.(Normal(mu1,sigma1),x1)))*pdf.(Normal(mu1,sigma1),x1),x))
    #plot(pdf.(Normal(mu1,sigma1),x1))

    gausswidth = 8*SD;  # 2.5 is the default for the function gausswin
    F = pdf.(Normal(gausswidth/2, SD),1:gausswidth);
    F /= sum(F);
        
    #try
        
        shift = Int(floor(length(F)/2)); # this is the amount of time that must be added to the beginning and end;
        
        if pad == "zeros"
            prefilt = vcat(zeros(shift,size(x,2)),x,zeros(shift,size(x,2)));
        elseif pad == "mean"
            #pads the beginning and end with copies of first and last value (not zeros)
            prefilt = vcat(broadcast(+,zeros(shift,size(x,2)),mean(x[1:SD,:],1)),x,
                broadcast(+,zeros(shift,size(x,2)),mean(x[end-SD:end,:],1)));        
        end
        
        postfilt = filt(F,prefilt); # filters the data with the impulse response in Filter
        postfilt = postfilt[2*shift:size(postfilt,1)-1,:];
        
    #catch
        
    #    postfilt = x
        
    #end
    

end

function psth(data::Dict,filt_sd::Float64,dt::Float64)

    #b = (1/3)* ones(1,3);
    #rate(data(j).N,1:size(data(j).spike_counts,1),j) = filter(b,1,data(j).spike_counts/dt)';
    
    try
        lambda = (1/dt) * FilterSpikes(filt_sd,data["spike_counts"]);
    catch
        lambda = (1/dt) * data["spike_counts"]/dt;
    end

end

function decimate(x, r)

    # Decimation reduces the original sampling rate of a sequence
    # to a lower rate. It is the opposite of interpolation.
    #
    # The decimate function lowpass filters the input to guard
    # against aliasing and downsamples the result.
    #
    #   y = decimate(x,r)
    #
    # Reduces the sampling rate of x, the input signal, by a factor
    # of r. The decimated vector, y, is shortened by a factor of r
    # so that length(y) = ceil(length(x)/r). By default, decimate
    # uses a lowpass Chebyshev Type I IIR filter of order 8.
    #
    # Sometimes, the specified filter order produces passband
    # distortion due to roundoff errors accumulated from the
    # convolutions needed to create the transfer function. The filter
    # order is automatically reduced when distortion causes the
    # magnitude response at the cutoff frequency to differ from the
    # ripple by more than 1E–6.

    nfilt = 8
    cutoff = .8 / r
    rip = 0.05  # dB

    function filtmag_db(b, a, f)
    # Find filter's magnitude response in decibels at given frequency.
    nb = length(b)
    na = length(a)
    top = dot(exp.(-1im*collect(0:nb-1)*pi*f), b)
    bot = dot(exp.(-1im*collect(0:na-1)*pi*f), a)
    20*log10(abs(top/bot))
    end        

    b, a = cheby1(nfilt, rip, cutoff)

    while all(b==0) || (abs(filtmag_db(b, a, cutoff)+rip)>1e-6)
    nfilt = nfilt - 1
    nfilt == 0 ? break : nothing
    b, a = cheby1(nfilt, rip, cutoff)
    end

    y = filtfilt(PolynomialRatio(b, a), x)
    nd = length(x)
    nout = ceil(nd/r)
    nbeg = Int(r - (r * nout - nd))
    y[nbeg:r:nd]
    
end

function cheby1(n, r, wp)

    # Chebyshev Type I digital filter design.
    #
    #    b, a = cheby1(n, r, wp)
    #
    # Designs an nth order lowpass digital Chebyshev filter with
    # R decibels of peak-to-peak ripple in the passband.
    #
    # The function returns the filter coefficients in length
    # n+1 vectors b (numerator) and a (denominator).
    #
    # The passband-edge frequency wp must be 0.0 < wp < 1.0, with
    # 1.0 corresponding to half the sample rate.
    #
    #  Use r=0.5 as a starting point, if you are unsure about choosing r.

    h = digitalfilter(Lowpass(wp), Chebyshev1(n, r))
    tf = convert(PolynomialRatio, h)
    coefb(tf), coefa(tf)

end

#function downsample_spiketrain(x;dtMC::Float64=1e-4,dt::Float64=2e-2)
#    
#    nbins = Int(dt/dtMC)    
#    x = vcat(x, mean(x[end-rem(length(x),nbins)+1:end]) * ones(eltype(x),mod(nbins - rem(length(x),nbins),nbins)))
#    x = squeeze(mean(reshape(x,nbins,:),1),1)  
#    
#    return x
#    
#end
