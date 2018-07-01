module analysis_functions

using DSP, Distributions, module_DDM_v4, StatsBase

export FilterSpikes, sample_latent, make_spikes, sampled_dataset!, diffLR

function FilterSpikes(x,SD)
    
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
        prefilt = cat(1,broadcast(+,zeros(shift,size(x,2)),mean(x[1:SD,:],1)),x,
        broadcast(+,zeros(shift,size(x,2)),mean(x[end-SD:end,:],1))); # pads the beginning and end with copies of first and last value (not zeros)
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

function sample_latent(T::Float64,L::Union{Float64,Vector{Float64}},R::Union{Float64,Vector{Float64}},
        x::Vector{Float64};dt::Float64=1e-4,path::Bool=false)
    
    vari, inatt, B, lambda, vara, vars, phi, tau_phi = x;
    
    nT = Int(ceil.(T/dt)); # number of timesteps
    
    path ? A = Vector{Float64}(nT) : nothing
                
    if rand() < inatt

        a = B * sign(randn())
        path ? A[1:nT] = a : nothing

    else

        La, Ra = make_adapted_clicks(L, R, phi, tau_phi);
        t = 0.:dt:nT*dt-dt; 
        hereL = vec(qfind(t,L))
        hereR = vec(qfind(t,R))
          
        a = sqrt(vari)*randn()
        t = 1

        while t <= nT

            #inputs
            any(t .== hereL) ? sL = sum(La[t .== hereL]) : sL = zero(phi)
            any(t .== hereR) ? sR = sum(Ra[t .== hereR]) : sR = zero(phi)
            var = vars * (sL + sR)  
            mu = -sL + sR
            var > zero(vars) ? a += mu + sqrt(var) * randn() : nothing

            #drift and diffuse
            a += (dt*lambda) * a + sqrt(vara * dt) * randn();

            abs(a) > B ? (a = B * sign(a); (path ? A[t:nT] = a : nothing); break) : ((path ? A[t] = a :nothing); t += 1)

        end               
    end
    
    path ? (return A) : (return a)
    
end

function sampled_dataset!(data,p,N;dtMC::Float64=1e-4,dt::Float64=2e-2)

    px = p[1:8]
    bias = p[9]
    #neural tuning curve parameters
    py = reshape(p[10:end],N,4)

    #sample latent paths
    A_sampled = pmap((T,leftbups,rightbups) -> sample_latent(T,leftbups,rightbups,px,path=true),
        data["T"],data["leftbups"],data["rightbups"]);
    #compute firing rates
    lambda_sampled = pmap((a,N) -> my_sigmoid(a,py[N,:]),A_sampled,data["N"]);
    #generate spikes
    spikes_sampled = pmap(lambda -> make_spikes(lambda),lambda_sampled);
    
    nbins = Int(dt/dtMC)

    #bin and downsample spikes
    for i = 1:length(spikes_sampled)
        spikes_sampled[i] = cat(1,spikes_sampled[i],
                zeros(nbins - rem(size(spikes_sampled[i],1),nbins),
                size(spikes_sampled[i],2)))
        data["spike_counts"][i] = squeeze(sum(reshape(spikes_sampled[i],nbins,
                        Int(size(spikes_sampled[i],1)/nbins),size(spikes_sampled[i],2)),1),1)
    end
    
    #make new choices
    for i = 1:length(data["pokedR"])
        data["pokedR"][i] = A_sampled[i][end] >= bias
    end
    
end

function make_spikes(lambda;dt::Float64=1e-4)
    
    spikes = copy(lambda)
    
    for i = 1:length(lambda)
        spikes[i] = Int(rand(Poisson(lambda[i]*dt)))
    end
    
    return spikes
end

function diffLR(nT,L,R;cumsum=false)
    #compute the cumulative diff of clicks
    t = 0:dt:nT*dt;
    L = fit(Histogram,L,t,closed=:left)
    R = fit(Histogram,R,t,closed=:left)
    
    if cumsum
        diffLR = cumsum(-L.weights + R.weights)
    else
        diffLR = sum(-L.weights + R.weights)
    end
    
end



end
