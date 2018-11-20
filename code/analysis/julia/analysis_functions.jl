module analysis_functions

using DSP, Distributions

export FilterSpikes, nanmean, nanstderr, rate_mat_func_filt

nanmean(x) = mean(filter(!isnan,x))
nanmean(x,y) = mapslices(nanmean,x,y)

nanstderr(x) = std(filter(!isnan,x))/sqrt(length(filter(!isnan,x)))
nanstderr(x,y) = mapslices(nanstderr,x,y)

function rate_mat_func_filt(nspikes,dt,filtSD)
    
    nT = map(length,nspikes)
    λ = fill!(Array{Float64,2}(length(nspikes),maximum(nT)),NaN);
    
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
        
    try
        
        shift = Int(floor(length(F)/2)); # this is the amount of time that must be added to the beginning and end;
        
        if pad == "zeros"
            prefilt = cat(1,zeros(shift,size(x,2)),x,zeros(shift,size(x,2)));
        elseif pad == "mean"
            #pads the beginning and end with copies of first and last value (not zeros)
            prefilt = cat(1,broadcast(+,zeros(shift,size(x,2)),mean(x[1:SD,:],1)),x,
                broadcast(+,zeros(shift,size(x,2)),mean(x[end-SD:end,:],1)));        
        end
        
        postfilt = filt(F,prefilt); # filters the data with the impulse response in Filter
        postfilt = postfilt[2*shift:size(postfilt,1)-1,:];
        
    catch
        
        postfilt = x
        
    end

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

end
