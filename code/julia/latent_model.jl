module latent_model

const dt,dimz = 2e-2,8

using global_functions

export LL_all_trials, ll_wrapper, dt, make_adapted_clicks, prior_on_spikes, bins, LL_single_trial, P0, Mprime!

function ll_wrapper{TT}(p_opt::Vector{TT}, p_const::Vector{Float64}, fit_vec::BitArray{1}, 
        data::Dict, model_type::Union{String,Array{String}};
        map_str::String="exp",n::Int=203,
        beta_y::Vector{Float64}=Vector{Float64}(0), 
        mu0_y::Vector{Vector{Float64}}=Vector{Vector{Float64}}(0),noise::String="Poisson")
    
    if any(model_type .== "spikes") & any(model_type .== "choice")
 
        pz,py,bias = latent_and_spike_params(p_opt, p_const, fit_vec, model_type)
        
        pz = map_latent_params!(pz,map_str,dt)   
        py = map(x->map_sig_params!(x,map_str),py)

        LL = LL_all_trials(pz, data, model_type, n=n, py=py, bias=bias)
        
        LLprior = prior_on_spikes(py,mu0_y,beta_y)
        
        LLprime = -(sum(LL) - LLprior)
        
    elseif any(model_type .== "spikes") 
        
        pz,py = latent_and_spike_params(p_opt, p_const, fit_vec, model_type)

        pz = map_latent_params!(pz,map_str,dt)   
        py = map(x->map_sig_params!(x,map_str),py)
        
        LL = LL_all_trials(pz, data, model_type, n=n, py=py, noise=noise)
        
        LLprior = prior_on_spikes(py,mu0_y,beta_y)
        
        LLprime = -(sum(LL) - LLprior)

    elseif any(model_type .== "choice")
        
        pz,bias = latent_and_spike_params(p_opt, p_const, fit_vec, model_type)
        
        pz = map_latent_params!(pz,map_str,dt)   
        LL = LL_all_trials(pz, data, model_type, n=n, bias=bias)
        
        LLprime = -sum(LL)
        
    end
    
    return LLprime
          
end

#perhaps at some point I should change this to "Gaussian prior" function?

function prior_on_spikes{TT}(py::Vector{Vector{TT}}, mu0_y::Vector{Vector{Float64}},
        beta_y::Vector{Float64})
    
        sum(map((p,mu0_y) -> sum(beta_y .* (p - mu0_y).^2),py,mu0_y))
    
end

function LL_all_trials{TT}(pz::Vector{TT}, 
        data::Dict, model_type::Union{String,Array{String}}; 
        comp_posterior::Bool=false, n::Int=203, 
        py::Union{Vector{Vector{TT}},Vector{Vector{Float64}}} = Vector{Vector{TT}}(0), 
        bias::Union{Float64,TT} = zero(TT),noise::String="Poisson")
        
    #break up latent variables
    vari,inatt,B,lambda_z,vara = pz[1:5]
   
    # spatial bin centers, width and edges
    xc,dx,xe = bins(B,n=n)
    
    # make initial latent distribution
    P = P0(vari,inatt,n,dx,xc);
   
    # build empty transition matrix
    M = zeros(TT,n,n);
    # build state transition matrix for no input time bins
    Mprime!(M,vara*dt,lambda_z,zero(TT),dx,xc,n=n)

    #compute bin location of bias and fraction of that bin that should go L or R
    if any(model_type .== "choice") 
        nbinsL, Sfrac = bias_bin(bias,xe,dx,n)
    end

    #compute expected firing rate for every neuron at every spatial bin
    any(model_type .== "spikes") ? lambda = map(p->my_sigmoid(xc,p),py) : nothing
    
    if any(model_type .== "spikes") & any(model_type .== "choice")
        
        output = pmap((L,R,T,nL,nR,N,SC,pokedR) -> LL_single_trial(pz, P, M, dx, xc, model_type,
            L, R, T, nL, nR,
            lambda=lambda[N],spike_counts=SC,
            nbinsL=nbinsL,Sfrac=Sfrac,pokedR=pokedR,
            comp_posterior=comp_posterior,n=n),
            data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"], 
            data["binned_rightbups"],
            data["N"],data["spike_counts"],
            data["pokedR"])
            
    elseif any(model_type .== "choice") 
        
        output = pmap((L,R,T,nL,nR,pokedR) -> LL_single_trial(pz, P, M, dx, xc, model_type,
            L, R, T, nL, nR,
            nbinsL=nbinsL,Sfrac=Sfrac,pokedR=pokedR,
            comp_posterior=comp_posterior,n=n),
            data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"], 
            data["binned_rightbups"],
            data["pokedR"])
        
    elseif any(model_type .== "spikes")
        
        output = pmap((L,R,T,nL,nR,N,SC) -> LL_single_trial(pz, P, M, dx, xc, model_type,
            L, R, T, nL, nR,
            lambda=lambda[N],spike_counts=SC,
            comp_posterior=comp_posterior,n=n,noise=noise),
            data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"], 
            data["binned_rightbups"],
            data["N"],data["spike_counts"])        
    else
        
        output = pmap((L,R,T,nL,nR) -> LL_single_trial(pz, P, M, dx, xc, model_type,
            L, R, T, nL, nR,
            comp_posterior=comp_posterior,n=n),
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
        comp_posterior::Bool=false, n::Int=203, noise::String="Poisson")
    
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
        if noise == "Poisson"
            Py = exp.(sum(map((k,lambda)->broadcast(-,broadcast(-,broadcast(*,k,log.(lambda'*dt)),lambda'*dt),
                lgamma.(k + 1)),spike_counts,lambda)))';
        elseif noise == "Gaussian"
            Py = exp.(sum(map((x,mu)->broadcast(-,log(1/sqrt(2*pi*0.1^2)),
                broadcast(/,(broadcast(-,x,mu')).^2,2*0.1^2)),spike_counts,lambda)))';
        end
    end
    
    c = Vector{TT}(T)
    comp_posterior ? alpha = Array{Float64,2}(n,T) : nothing

    @inbounds for t = 1:T
        
        any(t .== hereL) ? sL = sum(La[t .== hereL]) : sL = zero(phi)
        any(t .== hereR) ? sR = sum(Ra[t .== hereR]) : sR = zero(phi)

        var = vars * (sL + sR);  mu = -sL + sR

        ((sL + sR) > zero(vars)) ? (Mprime!(F,var+vara*dt,lambda_z,mu/dt,dx,xc,n=n); P  = F * P;) : P = M * P
        
        any(model_type .== "spikes") && (P .*= Py[:,t])
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

            any(model_type .== "spikes") && (P .*= Py[:,t+1])
            any(model_type .== "choice") && t + 1 == T && (P .*=  Pd)
        
            any(t+1 .== hereL) ? sL = sum(La[t+1 .== hereL]) : sL = zero(phi)
            any(t+1 .== hereR) ? sR = sum(Ra[t+1 .== hereR]) : sR = zero(phi)

            var = vars * (sL + sR);  mu = -sL + sR

            (var > zero(vars)) ? (Mprime!(F,var+vara*dt,lambda_z,mu/dt,dx,xc,n=n); P  = F' * P;) : P = M' * P

            P /= c[t+1] 

            beta[:,t] = P

        end

    end

    comp_posterior ? (return alpha .* beta) : (return sum(log.(c)))

end

function bins{TT}(B::TT;n::Int=203)
    
    dx = 2.*B/(n-2);  #bin width
    xc = vcat(collect(linspace(-(B+dx/2.),-dx,(n-1)/2.)),0.,collect(linspace(dx,(B+dx/2.),(n-1)/2))); #centers
    xe = cat(1,xc[1]-dx/2,xc+dx/2) #edges
    
    return xc, dx, xe
    
end

function P0{TT}(vari::TT,inatt::TT,n::Int,dx::TT,xc::Vector{TT})
    
    # make initial delta function
    P = zeros(xc); 
    P[[1,n]] = inatt/2.; 
    P[ceil(Int,n/2)] = one(TT) - inatt; 
    # build empty transition matrix
    M = zeros(TT,n,n);
    Mprime!(M,vari,zero(TT),zero(TT),dx,xc,n=n); 
    P = M * P
    
end

function bias_bin{TT}(bias::TT,xe::Vector{TT},dx::TT,n::Int)
    
    #nbinsL = ceil(Int,(B+bias)/dx)
    #Sfrac = one(dx)/dx * (bias - (-(B+dx)+nbinsL*dx))
    nbinsL = sum(bias .> xe[2:n])
    Sfrac = (bias - xe[nbinsL+1])/dx
    Sfrac < zero(Sfrac) ? Sfrac = zero(Sfrac) : nothing
    Sfrac > one(Sfrac) ? Sfrac = one(Sfrac) : nothing
    
    return nbinsL, Sfrac
    
end

function Mprime!{TT}(F::Array{TT,2},vara::TT,lambda::TT,h::TT,dx::TT,xc::Vector{TT}; n::Int=203)
    
    F[1,1] = one(TT); F[n,n] = one(TT); F[:,2:n-1] = zero(TT)

    ndeltas = max(70,ceil(Int, 10.*sqrt(vara)/dx));

    (ndeltas > 1e3 && h == zero(TT)) ? (println(vara); println(dx); println(ndeltas)) : nothing

    #deltas = collect(-ndeltas:ndeltas) * (5.*sqrt(vara))/ndeltas;
    #ps = broadcast(exp, broadcast(/, -broadcast(^, deltas,2), 2.*vara)); ps = ps/sum(ps);
    
    deltaidx = collect(-ndeltas:ndeltas);
    deltas = deltaidx * (5.*sqrt(vara))/ndeltas;
    ps = exp.(-0.5 * (5*deltaidx./ndeltas).^2); ps = ps/sum(ps);
    
    @inbounds for j = 2:n-1

        abs(lambda) < 1e-150 ? mu = xc[j] + h * dt : mu = exp(lambda*dt)*(xc[j] + h/lambda) - h/lambda
        
        #now we're going to look over all the slices of the gaussian
        for k = 1:2*ndeltas+1

            s = mu + deltas[k]

            if s <= xc[1]

                F[1,j] += ps[k];

            elseif s >= xc[n]

                F[n,j] += ps[k];

            else

                if xc[1] < s && xc[2] > s

                    lp,hp = 1,2;

                elseif xc[n-1] < s && xc[n] > s

                    lp,hp = n-1,n;

                else

                    hp,lp = ceil(Int, (s-xc[2])/dx) + 2, floor(Int, (s-xc[2])/dx) + 2;

                end

                if (hp == lp)

                    F[lp,j] += ps[k];

                else

                    dd = xc[hp] - xc[lp];
                    F[hp,j] += ps[k]*(s-xc[lp])/dd;
                    F[lp,j] += ps[k]*(xc[hp]-s)/dd;

                end

            end

        end

    end

end

end
