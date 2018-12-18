module latent_model

const dimz = 8

using latent_DDM_common_functions

export LL_all_trials, ll_wrapper, prior_on_spikes, LL_single_trial

function ll_wrapper{TT}(p_opt::Vector{TT}, p_const::Vector{Float64}, fit_vec::BitArray{1}, 
        data::Dict, model_type::Union{String,Array{String}}, f_str::String;
        map_str::String="exp",n::Int=203,
        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(0), 
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}(0),noise::String="Poisson",dt::Float64=2e-2)
    
    if any(model_type .== "spikes") & any(model_type .== "choice")
 
        pz,py,bias = latent_and_spike_params(p_opt, p_const, fit_vec, model_type)
        
        pz = map_latent_params!(pz,map_str,dt)   
        #py = map(x->map_sig_params!(x,map_str),py)
        py = map(x->map_lambda_y_p!(x,f_str;map_str=map_str),py)

        LL = LL_all_trials(pz, data, model_type, n=n, py=py, bias=bias, dt=dt)
        
        LLprior = prior_on_spikes(py,mu0,beta)
        
        LLprime = -(sum(LL) - LLprior)
        
    elseif any(model_type .== "spikes") 
        
        pz,py = latent_and_spike_params(p_opt, p_const, fit_vec, model_type, f_str)

        pz = map_latent_params!(pz,map_str,dt)       
        py = map(x->map_lambda_y_p!(x,f_str;map_str=map_str),py)
        
        #if f_str == "exp"
        #    py = map(x->map_exp_params!(x),py)
        #elseif f_str == "sig"
        #    py = map(x->map_sig_params!(x,map_str),py);
        #end
        
        LL = LL_all_trials(pz, data, model_type, f_str, n=n, py=py, noise=noise, dt=dt)
        
        LLprior = prior_on_spikes(py,mu0,beta)
        
        LLprime = -(sum(LL) - LLprior)

    elseif any(model_type .== "choice")
        
        pz,bias = latent_and_spike_params(p_opt, p_const, fit_vec, model_type)
        
        pz = map_latent_params!(pz,map_str,dt)   
        LL = LL_all_trials(pz, data, model_type, n=n, bias=bias,dt=dt)
        
        LLprime = -sum(LL)
        
    end
    
    return LLprime
          
end

#perhaps at some point I should change this to "Gaussian prior" function?

function prior_on_spikes{TT}(py::Vector{Vector{TT}}, mu0::Vector{Vector{Float64}},
        beta::Vector{Vector{Float64}})
    
        sum(map((p,mu0,beta) -> sum(beta .* (p - mu0).^2),py,mu0,beta))
    
end

function LL_all_trials{TT}(pz::Vector{TT}, 
        data::Dict, model_type::Union{String,Array{String}}, f_str::String; 
        comp_posterior::Bool=false, n::Int=203, 
        py::Union{Vector{Vector{TT}},Vector{Vector{Float64}}} = Vector{Vector{TT}}(0), 
        bias::Union{Float64,TT} = zero(TT),noise::String="Poisson", dt::Float64=2e-2)
        
    #break up latent variables
    vari,inatt,B,lambda_z,vara = pz[1:5]
   
    # spatial bin centers, width and edges
    xc,dx,xe = bins(B,n=n)
    
    # make initial latent distribution
    P = P0(vari,inatt,n,dx,xc);
   
    # build empty transition matrix
    M = zeros(TT,n,n);
    # build state transition matrix for no input time bins
    Mprime!(M,vara*dt,lambda_z,zero(TT),dx,xc,n=n,dt=dt)

    #compute bin location of bias and fraction of that bin that should go L or R
    if any(model_type .== "choice") 
        nbinsL, Sfrac = bias_bin(bias,xe,dx,n)
    end

    #compute expected firing rate for every neuron at every spatial bin
    #any(model_type .== "spikes") ? lambda = map(p->my_sigmoid(xc,p),py) : nothing
    if any(model_type .== "spikes")
        lambda = map(p->lambda_y(xc,p,f_str),py)
        #if f_str == "exp"
        #    lambda = map(p->my_exp(xc,p),py)
        #elseif f_str == "sig"
        #    lambda = map(p->my_sigmoid(xc,p),py)
        #end
    end
    
    if any(model_type .== "spikes") & any(model_type .== "choice")
        
        output = pmap((L,R,T,nL,nR,N,SC,pokedR) -> LL_single_trial(pz, P, M, dx, xc, model_type,
            L, R, T, nL, nR,
            lambda=lambda[N],spike_counts=SC,
            nbinsL=nbinsL,Sfrac=Sfrac,pokedR=pokedR,
            comp_posterior=comp_posterior,n=n,dt=dt),
            data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"], 
            data["binned_rightbups"],
            data["N"],data["spike_counts"],
            data["pokedR"])
            
    elseif any(model_type .== "choice") 
        
        output = pmap((L,R,T,nL,nR,pokedR) -> LL_single_trial(pz, P, M, dx, xc, model_type,
            L, R, T, nL, nR,
            nbinsL=nbinsL,Sfrac=Sfrac,pokedR=pokedR,
            comp_posterior=comp_posterior,n=n,dt=dt),
            data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"], 
            data["binned_rightbups"],
            data["pokedR"])
        
    elseif any(model_type .== "spikes")
        
        output = pmap((L,R,T,nL,nR,N,SC) -> LL_single_trial(pz, P, M, dx, xc, model_type,
            L, R, T, nL, nR,
            lambda=lambda[N],spike_counts=SC,
            comp_posterior=comp_posterior,n=n,noise=noise,dt=dt),
            data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"], 
            data["binned_rightbups"],
            data["N"],data["spike_counts"])        
    else
        
        output = pmap((L,R,T,nL,nR) -> LL_single_trial(pz, P, M, dx, xc, model_type,
            L, R, T, nL, nR,
            comp_posterior=comp_posterior,n=n,dt=dt),
            data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"], 
            data["binned_rightbups"])

    end
    
end

function LL_single_trial{TT}(pz::Vector{TT}, P::Vector{TT}, M::Array{TT,2}, dx::TT, 
        xc::Vector{TT}, model_type::Union{String,Array{String}},
        L::Vector{Float64}, R::Vector{Float64}, T::Int,
        hereL::Vector{Int}, hereR::Vector{Int};
        lambda::Vector{Vector{TT}}=Vector{Vector{TT}}(0),
        spike_counts::Union{Vector{Vector{Float64}},Vector{Vector{Int}}}=Vector{Vector{Float64}}(0),
        nbinsL::Union{TT,Int}=1, Sfrac::TT=one(TT), pokedR::Bool=false, 
        comp_posterior::Bool=false, n::Int=203, noise::String="Poisson", dt::Float64=2e-2,
        std::Vector{Float64}=Vector{Float64}(0))
    
    #break up parameters
    lambda_z,vara,vars,phi,tau_phi = pz[4:8]

    #adapt magnitude of the click inputs
    La, Ra = make_adapted_clicks(phi,tau_phi,L,R)

    #empty transition matrix for time bins with clicks
    F = zeros(M)

    #make choice observation distribution
    if any(model_type .== "choice")
        pokedL = convert(TT,!pokedR); pokedR = convert(TT,pokedR)
        Pd = vcat(pokedL * ones(nbinsL), pokedL * Sfrac + pokedR * (one(Sfrac) - Sfrac), pokedR * ones(n - (nbinsL + 1)))
    end
    
    #make spike count distribution
    if any(model_type .== "spikes") 
        #if noise == "Poisson"
            #Py = exp.(sum(map((k,lambda)->broadcast(-,broadcast(-,broadcast(*,k,log.(lambda'*dt)),lambda'*dt),
            #    lgamma.(k + 1)),spike_counts,lambda)))';
        #this is silly, should do this above, but didn't get around to it.
        N = length(spike_counts);
        spike_counts = reshape(vcat(spike_counts...),:,N);
        lambda = reshape(vcat(lambda...),n,:);            
        #elseif noise == "Gaussian"
        #    Py = exp.(sum(map((x,mu)->broadcast(-,log(1/sqrt(2*pi*1e-6^2)),
        #        broadcast(/,(broadcast(-,x,mu')).^2,2*1e-6^2)),spike_counts,lambda)))';
        #end
    end
    
    c = Vector{TT}(T)
    comp_posterior ? alpha = Array{Float64,2}(n,T) : nothing

    @inbounds for t = 1:T
        
        any(t .== hereL) ? sL = sum(La[t .== hereL]) : sL = zero(phi)
        any(t .== hereR) ? sR = sum(Ra[t .== hereR]) : sR = zero(phi)

        var = vars * (sL + sR);  mu = -sL + sR

        ((sL + sR) > zero(vars)) ? (Mprime!(F,var+vara*dt,lambda_z,mu/dt,dx,xc,n=n,dt=dt); P  = F * P;) : P = M * P
        
        if any(model_type .== "spikes")
            #Py = exp.(sum(map((k,lambda)->k*log.(lambda'*dt)-lambda'*dt-lgamma(k+1),spike_counts[t,:],lambda)))';
            #P .*= exp.((spike_counts[t,:]'*log.(lambda*dt)')' - sum(lambda,2)*dt - sum(lgamma.(spike_counts[t,:] + 1)));
            if noise == "Poisson"
                P .*= vec(exp.((spike_counts[t,:]'*log.(lambda*dt)')' - sum(lambda,2)*dt - sum(lgamma.(spike_counts[t,:] + 1))));
            elseif noise == "Gaussian"
                #noise is hard-coded in, should eventually used std (varargin)
                arg = sum(log.(1./sqrt.(2*pi*(5e0*ones(N)).^2))) - 
                        sum(broadcast(/,(broadcast(-,spike_counts[t,:],lambda')).^2,2*(5e0*ones(N)).^2)',2);
                #protect against NaNs in gradient
                arg[arg .< log(1e-150)] = log(1e-150);
                P .*= vec(exp.(arg));
            end
            #P .*= Py;
        end
        
        #any(model_type .== "spikes") && (P .*= Py[:,t])
        (any(model_type .== "choice") && t == T) && (P .*=  Pd)

        c[t] = sum(P)
        P /= c[t] 

        comp_posterior ? alpha[:,t] = P : nothing

    end

    if comp_posterior

        beta = zeros(Float64,n,T)
        P = ones(Float64,n); #initialze backward pass with all 1's
    
        beta[:,T] = P;

        @inbounds for t = T-1:-1:1
            
            if any(model_type .== "spikes")
                P .*= vec(exp.((spike_counts[t+1,:]'*log.(lambda*dt)')' - 
                        sum(lambda,2)*dt - sum(lgamma.(spike_counts[t+1,:] + 1))));
            end

            #any(model_type .== "spikes") && (P .*= Py[:,t+1])
            any(model_type .== "choice") && t + 1 == T && (P .*=  Pd)
        
            any(t+1 .== hereL) ? sL = sum(La[t+1 .== hereL]) : sL = zero(phi)
            any(t+1 .== hereR) ? sR = sum(Ra[t+1 .== hereR]) : sR = zero(phi)

            var = vars * (sL + sR);  mu = -sL + sR

            (var > zero(vars)) ? (Mprime!(F,var+vara*dt,lambda_z,mu/dt,dx,xc,n=n,dt=dt); P  = F' * P;) : P = M' * P

            P /= c[t+1] 

            beta[:,t] = P

        end

    end

    comp_posterior ? (return alpha .* beta) : (return sum(log.(c)))

end


function map_pz!(x,map_str,dt)
    
    x[[1,5,6]] = exp.(x[[1,5,6]]);
    x[2] = 0.5*(1+tanh(x[2]));
    
    if map_str == "exp"
        x[3] = 2. + exp(x[3]);
    elseif map_str == "tanh"
        x[3] = 2 + 100 * 0.5*(1+tanh.(x[3]))
    end
    
    x[4] = -1./(2*dt) + (1./dt)*(0.5*(1.+tanh(x[4])));
    x[7] = exp(x[7]);
    x[8] = exp(x[8]);
    
    return x
    
end

function inv_map_pz!(x,map_str,dt)
    
    x[[1,5,6]] = log.(x[[1,5,6]]);
    x[2] = atanh(2.*x[2]-1.);
    
    if map_str == "exp"
        x[3] = log(x[3]-2.);
    elseif map_str == "tanh"
        x[3] = atanh.(((x[3] - 2.)/(100*0.5))-1)
    end
    
    x[4] = atanh((2 .* dt * (x[4] + 1./(2.*dt))) - 1.);
    x[7] = log(x[7]);
    x[8] = log(x[8]);
    
    return x
    
end

function latent_and_spike_params{TT}(p_opt::Vector{TT}, p_const::Vector{Float64}, fit_vec::BitArray{1}, model_type::String, 
    f_str::String)
    
    p = Vector{TT}(length(fit_vec))
    p[fit_vec] = p_opt;
    p[.!fit_vec] = p_const;
    
    if any(model_type .== "spikes") & any(model_type .== "choice")
 
        pz = p[1:8];
        bias = p[9];
        
        if f_str == "sig"
            
            pytemp = reshape(p[10:end],4,:)
            
        elseif f_str == "exp"
            
            pytemp = reshape(p[10:end],2,:)
            
        elseif f_str == "softplus"
            
            pytemp = reshape(p[10:end],3,:)
            
        end
        
        py = Vector{Vector{TT}}(0)

        #py = map(i->pytemp[:,i],size(pytemp,2))
        for i = 1:size(pytemp,2)
            push!(py,pytemp[:,i])
        end
               
        return pz,py,bias
    
    elseif any(model_type .== "spikes") 
                
        pz = p[1:8];

        if f_str == "sig"
        
            pytemp = reshape(p[9:end],4,:)

        elseif f_str == "exp"
            
            pytemp = reshape(p[9:end],2,:)
            
        elseif f_str == "softplus"
            
            pytemp = reshape(p[9:end],3,:)
            
        end
        
        py = Vector{Vector{TT}}(0)

        #py = map(i->pytemp[:,i],size(pytemp,2))
        for i = 1:size(pytemp,2)
            push!(py,pytemp[:,i])
        end

        return pz, py

    elseif any(model_type .== "choice")
                
        pz = p[1:8];
        bias = p[9];
                
        return pz, bias
        
    end
    
end

end
