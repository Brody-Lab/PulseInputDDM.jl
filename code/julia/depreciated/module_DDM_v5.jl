module module_DDM_v5

const dt,dims = 2e-2,Dict("z"=>8,"d"=>1,"y"=>4)

import ForwardDiff, Base.convert
using StatsBase, LsqFit, Distributions

export LL_all_trials, LL_single_trial, convert_data!
export deparameterize, reparameterize, my_sigmoid, Mprime!, make_adapted_clicks
export ll_wrapper, fit_func, qfind, package_data!, compute_x0, sample_latent
export dt,dims

convert(::Type{Float64}, x::ForwardDiff.Dual) = Float64(x.value)
convert(::Type{Int},x::ForwardDiff.Dual) = Int(x.value)

function my_sigmoid(x,p)
    
    y = broadcast(+,p[:,1]',broadcast(/,p[:,2]',(1. + exp.(broadcast(+,broadcast(*,-p[:,3]',x),p[:,4]')))));
    
end

function compute_x0(data,model_type,N)

    x0 = [1e-5, 0., 20., 1e-3, 10., 1., 1.-1e-6, 0.2]

   if any(model_type .== ["choice","joint"])
        x0 = cat(1,x0,0.)
    end

    if any(model_type .== ["spikes","joint"])
        tri = length(data["T"]);
        temp = Vector{Array{Float64,2}}(N)

        for i = 1:N
            temp[i] = Array{Float64,2}(0,2)
        end

        #loop over trials
        for i = 1:tri
             
            #compute the cumulative diff of clicks
            t = 0:dt:data["nT"][i]*dt;
            L = fit(Histogram,data["leftbups"][i],t,closed=:left)
            R = fit(Histogram,data["rightbups"][i],t,closed=:left)
            diffLR = cumsum(-L.weights + R.weights)
                                        
            for j = 1:length(data["N"][i])
                temp[data["N"][i][j]] = cat(1,temp[data["N"][i][j]],
                                            cat(2,diffLR,data["spike_counts"][i][:,j]/dt))
            end
                                                             
        end

        model(x,p) = p[1]^2 + p[2]^2./(1. + exp.(-p[3] .* x + p[4]))

        p0 = [0.,10.,0.,0.];
        x0y = Array{Float64,2}(N,4)

        for j = 1:N
            fit = curve_fit(model, temp[j][:,1], temp[j][:,2], p0)
            x0y[j,:] = fit.param
            x0y[j,1:2] = x0y[j,1:2].^2 + eps();
        end

        x0 = cat(1,x0,vec(x0y))

    end

    return  x0

end

function qfind(x,ts)

    # function y=qfind(x,ts)
    # x is a vector , t is the target (can be one or many targets),
    # y is same length as ts
    # does a binary search: assumes that x is sorted low to high and unique.

    ys = zeros(Int,size(ts));

    for i = 1:length(ts)

        t = ts[i];

        if isnan(t)
            y = NaN;
        else

            high = length(x)::Int;
            low = -1;

            if t >= x[end]
                y = length(x)::Int;
            else

                try
                    while (high - low > 1)

                        probe = Int(ceil((high + low) / 2));

                        if x[probe] > t
                            high = probe;
                        else
                            low = probe;
                        end

                    end
                    
                    y = low;

                catch

                    y = low;

                end
            end
        end
        
        ys[i] = y;

    end

    return ys

end

function package_data!(data,rawdata,model_type,t0,N0)

    ntrials = length(rawdata["T"])

    append!(data["T"],rawdata["T"])
    append!(data["nT"],ceil.(Int,rawdata["T"]/dt))
    append!(data["pokedR"],vec(convert(BitArray,rawdata["pokedR"])))
    append!(data["correct_dir"],vec(convert(BitArray,rawdata["correct_dir"])))

    for i = 1:ntrials
        t = 0.:dt:data["nT"][t0 + i]*dt;
        push!(data["leftbups"],vec(collect(rawdata["leftbups"][i]))) #this will be a scalar or a vector, vec(collect()) will ensure that it is a vector in either case
        push!(data["rightbups"],vec(collect(rawdata["rightbups"][i])))
        push!(data["hereL"], vec(qfind(t,rawdata["leftbups"][i]))) #qfind always returns an array, so vec will turn into a vector
        push!(data["hereR"], vec(qfind(t,rawdata["rightbups"][i])))
    end

    if any(model_type .== ["spikes","joint"])

        N = size(rawdata["St"][1],2)

        for i = 1:ntrials

            t = 0.:dt:data["nT"][t0 + i]*dt;
            temp = Array{Int}(data["nT"][t0 + i],0)
            tempN = Array{Int}(0);

            for j = 1:N

                if size(rawdata["St"][i][j],2) > 0

                    length(rawdata["St"][i][j]) == 1 ? temp2 = vec(collect(rawdata["St"][i][j])) : temp2 = vec(rawdata["St"][i][j])

                    results = fit(Histogram,temp2,t,closed=:left)
                    temp = hcat(temp,results.weights)
                    tempN = vcat(tempN,N0+j)

                end
            end

            push!(data["spike_counts"],temp)
            push!(data["N"], tempN)
        end

        N0 += N

    end

    if model_type == "choice"
        for i = 1:ntrials
            rawdata["N"][i] = Array{Int,1}(0,);
            rawdata["spike_counts"][i] = Array{Int,2}(0,2)
        end
    end

    t0 += ntrials

    return data, t0, N0

end

function fit_func(model_type::String,N::Int)

    #          vari       inatt          B    lambda       vara    vars     phi    tau_phi 
    fit_vec = [falses(1);falses(1);    trues(4);                         trues(2)];

    if any(model_type .== ["choice","joint"]);
        fit_vec = cat(1,fit_vec,trues(1));
    end
    if any(model_type .== ["spikes","joint"])
        fit_vec = cat(1,fit_vec,trues(dims["y"]*N));
    end

    return fit_vec

end

function deparameterize{TT}(xf::Vector{TT}, x0::Vector{Float64}, fit_vec::BitArray{1},model_type::String, N::Int64)

    x = Array{TT}(length(fit_vec))
    x[fit_vec] = xf;
    x[.!fit_vec] = x0[.!fit_vec];

    x[[1,5,6]] = sqrt.(x[[1,5,6]]);
    x[2] = atanh(2.*x[2]-1.);
    x[3] = sqrt(x[3]-2.);
    x[4] = atanh((2 .* dt * (x[4] + 1./(2.*dt))) - 1.);
    x[7] = sqrt(x[7] - 0.1);
    x[8] = sqrt(x[8] - 0.02);

    if model_type == "choice"
        
        x[9] = atanh(2.*x[9] - 1.);
        
    elseif model_type == "spikes"
        
        x[8+1:8+2*N] = sqrt.(x[8+1:8+2*N]);
        
    elseif model_type == "joint"
        
        x[9] = atanh(2.*x[9] - 1.);
        x[9+1:9+2*N] = sqrt.(x[9+1:9+2*N]);
        
    end

    return x

end

function reparameterize{TT}(xf::Vector{TT}, x0::Vector{Float64}, fit_vec::BitArray{1}, model_type::String, N::Int64)

    x = Array{TT}(length(fit_vec))
    x[fit_vec] = xf;
    x[.!fit_vec] = x0[.!fit_vec];

    x[[1,5,6]] = x[[1,5,6]].^2;
    x[2] = 0.5*(1+tanh(x[2]));
    x[3] = 2 + x[3]^2;
    x[4] = -1./(2*dt) + (1./dt)*(0.5*(1.+tanh(x[4])));
    x[7] = 0.1 + x[7]^2;
    x[8] = 0.02 + x[8]^2;

    if model_type == "choice"
        
        x[9] = 0.5*(1 + tanh(x[9]));
        
    elseif model_type ==  "spikes"
        
        x[8+1:8+2*N] = x[8+1:8+2*N].^2;
        
    elseif model_type == "joint"

        x[9] = 0.5*(1 + tanh(x[9]));
        x[9+1:9+2*N] = x[9+1:9+2*N].^2;
        
    end

    return x

end

function ll_wrapper{TT}(xf::Vector{TT}, data::Dict, model_type::String, x0::Vector{Float64}, 
        fit_vec::BitArray{1}; N::Int64=0, n::Int=203,beta::Dict=Dict("d"=> 0.))

    x = reparameterize(xf,x0,fit_vec,model_type,N)
    LL = LL_all_trials(x,data,model_type,N=N,n=n,beta=beta)
    return LL

end

function convert_data!(data,model_type)

    data["T"] = convert(Array{Float64,2},data["T"])
    data["nT"] = convert(Array{Int,2},data["nT"])
    data["Ntotal"] = convert(Int,data["Ntotal"][1])

    for i = 1:length(data["hereL"])

        if any(model_type .== ["spikes","joint"])
            if length(data["N"][i]) > 1
                data["N"][i] = vec(convert(Array{Int,2},data["N"][i]));
            else
                data["N"][i] = convert(Int,data["N"][i])
            end

             if length(data["spike_counts"][i]) > 1
                data["spike_counts"][i] = convert(Array{Int,2},data["spike_counts"][i])
            else
                #this is a crazy way to deal with a situations when you have only one spike and one time bin
                data["spike_counts"][i] = convert(Array{Int,1},Float64[data["spike_counts"][i]])
            end

        else
            data["N"][i] = Array{Int,1}(0,)
            data["spike_counts"][i] = Array{Int,2}(0,2)
        end
        
        if length(data["hereL"][i]) > 1
            data["hereL"][i] = vec(convert(Array{Int},data["hereL"][i]))
        else
            data["hereL"][i] = convert(Int,data["hereL"][i])
        end

        if length(data["hereR"][i]) > 1
            data["hereR"][i] = vec(convert(Array{Int},data["hereR"][i]))
        else
            data["hereR"][i] = convert(Int,data["hereR"][i])
        end
    end

end

function make_adapted_clicks(leftbups, rightbups, phi, tau_phi)

    L = ones(typeof(phi),size(leftbups));
    R = ones(typeof(phi),size(rightbups));

    if phi !== 1.

    # magnitude of stereo clicks set to zero
    if ~isempty(leftbups) && ~isempty(rightbups) && abs(leftbups[1]-rightbups[1]) < eps()
        L[1] = eps()
        R[1] = eps()
    end

        if length(leftbups) <= 1
            ici_l = [];
        else
            ici_L = (leftbups[2:end]  - leftbups[1:end-1])'
        end

        if length(rightbups) <= 1
            ici_R = []
        else
            ici_R = (rightbups[2:end]  - rightbups[1:end-1])'
        end

        for i = 2:length(leftbups)
            if abs(1. - L[i-1]*phi) <= 1e-150
                L[i] = 1.
            else
                last_L = tau_phi*log(abs(1-L[i-1]*phi))
                L[i] = 1 - exp((-ici_L[i-1] + last_L)/tau_phi)
            end
        end;

        for i = 2:length(rightbups)
            if abs(1. - R[i-1]*phi) <= 1e-150
                R[i] = 1.
            else
                last_R = tau_phi*log(abs(1-R[i-1]*phi))
                R[i] = 1 - exp((-ici_R[i-1] + last_R)/tau_phi)
            end
        end;

    end

        L = real(L)
        R = real(R)

    return L, R

end

function LL_single_trial{TT}(L::Union{Array{Float64},Float64},R::Union{Array{Float64},Float64},T::Int,
        hereL::Union{Array{Int},Int},hereR::Union{Array{Int},Int}, 
        P::Vector{TT},lambda_drift::TT,vara::TT,vars::TT,phi::TT,tau_phi::TT,M::Array{TT,2},dx::TT, xc::Vector{TT},
        model_type::String;
        lambda::Union{Array{TT,2},Array{TT,1}}=Array{TT}(0,2),
        spike_counts::Union{Array{Int,1},Array{Int,2}}=Array{Int,2}(0,2),
        nbinsL::TT=one(TT),Sfrac::TT=one(TT),pokedR::Union{Bool,TT}=false,comp_posterior::Bool=false, n::Int=203)

    La, Ra = make_adapted_clicks(L,R,phi,tau_phi)

    F = zeros(M)

    if any(model_type .== ["choice","joint"])
        notpoked = convert(TT,~pokedR); poked = convert(TT,pokedR)
        Pd = vcat(notpoked * ones(nbinsL), notpoked * Sfrac + poked * (one(Sfrac) - Sfrac), poked * ones(n - (nbinsL + 1)))
    end
    
    if any(model_type .== ["spikes","joint"]) 
        Py = exp.(broadcast(-, broadcast(-, spike_counts *  log.(lambda'*dt), sum(lambda,2)' * dt), 
                sum(lgamma.(spike_counts + 1),2)))' 
    end
    
    c = Vector{TT}(T)
    comp_posterior ? alpha = Array{Float64,2}(n,T) : nothing

    @inbounds for t = 1:T
        
        any(t .== hereL) ? sL = sum(La[t .== hereL]) : sL = zero(phi)
        any(t .== hereR) ? sR = sum(Ra[t .== hereR]) : sR = zero(phi)

        var = vars * (sL + sR);  mu = -sL + sR

        (var > zero(vars)) ? (Mprime!(F,var+vara*dt,lambda_drift,mu/dt,dx,xc,n=n); P  = F * P;) : P = M * P
        
        any(model_type .== ["spikes","joint"]) && (P .*= Py[:,t])
        any(model_type .== ["choice","joint"]) && t == T && (P .*=  Pd)

        c[t] = sum(P)
        P /= c[t] 

        comp_posterior ? alpha[:,t] = P : nothing

    end

    if comp_posterior

        beta = zeros(Float64,n,T)
        P = ones(Float64,n); #initialze backward pass with all 1's
    
        beta[:,T] = P;

        @inbounds for t = T-1:-1:1

            any(model_type .== ["spikes","joint"]) && (P .*= Py[:,t+1])
            any(model_type .== ["choice","joint"]) && t + 1 == T && (P .*=  Pd)
        
            any(t+1 .== hereL) ? sL = sum(La[t+1 .== hereL]) : sL = zero(phi)
            any(t+1 .== hereR) ? sR = sum(Ra[t+1 .== hereR]) : sR = zero(phi)

            var = vars * (sL + sR);  mu = -sL + sR

            (var > zero(vars)) ? (Mprime!(F,var+vara*dt,lambda_drift,mu/dt,dx,xc,n=n); P  = F' * P;) : P = M' * P

            P /= c[t+1] 

            beta[:,t] = P

        end

    end

    comp_posterior ? (return alpha .* beta) : (return sum(log.(c)))

end

function group_params{TT}(xf::Vector{TT}, x0::Vector{Float64}, fit_vec::BitArray{1})
    
    x = Array{TT}(length(fit_vec))
    x[fit_vec] = xf;
    x[.!fit_vec] = x0[.!fit_vec];
    
    return x
    
end

function LL_all_trials{TT}(xf::Vector{TT}, x0::Vector{Float64}, fit_vec::BitArray{1},
        data::Dict, model_type::String;
        N::Int64=0, comp_posterior::Bool=false, n::Int=203, beta::Dict=Dict("d"=>0.))
    
    x = group_params(xf,x0,fit_vec)

    vari,inatt,B,lambda_drift,vara,vars,phi,tau_phi = x[1:dims["z"]]

    ntrials = length(data["T"])
   
    # binning
    dx = 2.*B/(n-2);  #bin width
    xc = vcat(collect(linspace(-(B+dx/2.),-dx,(n-1)/2.)),0.,collect(linspace(dx,(B+dx/2.),(n-1)/2)));
    
    # make initial delta function
    P = zeros(xc); P[[1,n]] = inatt/2.; P[ceil(Int,n/2)] = one(TT) - inatt; 
    # Convolve initial delta with vari
    M = zeros(TT,n,n);  Mprime!(M,vari,zero(TT),zero(TT),dx,xc,n=n); P = M * P
    #make ntrial copys for all the trials
    P = pmap(i->copy(P),1:ntrials); 
    
    # build state transition matrix for no input time bins
    Mprime!(M,vara*dt,lambda_drift,zero(TT),dx,xc,n=n)

    if any(model_type .== ["choice","joint"])
        bias = x[9]
        nbinsL = ceil(Int,(B+bias)/dx)
        Sfrac = one(dx)/dx * (bias - (-(B+dx)+nbinsL*dx))
    end

    if any(model_type .== ["spikes","joint"])
        model_type == "joint" ? xy = reshape(x[dims["z"]+dims["d"]+1:end],N,4) : xy = reshape(x[dims["z"]+1:end],N,4)

        lambda = my_sigmoid(xc,xy)
        temp = broadcast(+,broadcast(*,-xy[:,3]',xc),xy[:,4]')
        lambda[exp.(temp) .<= 1e-150] = broadcast(+,xy[:,1]',broadcast(/,xy[:,2]',ones(n,)))[exp.(temp) .<= 1e-150]
        lambda[exp.(temp) .>= 1e150] = broadcast(*,xy[:,1]',ones(n,))[exp.(temp) .>= 1e150]
    end
        
    output = pmap((L,R,T,nL,nR,P,N,SC) -> LL_single_trial(L,R,T,nL,nR,P,
            lambda_drift,vara,vars,phi,tau_phi,M,dx,xc,model_type,
            lambda=lambda[:,N],spike_counts=SC,comp_posterior=comp_posterior,n=n),
            data["leftbups"],data["rightbups"],data["nT"],data["hereL"],data["hereR"],P,
            data["N"],data["spike_counts"])

    comp_posterior ? (return output) : (return -(sum(output) - beta["d"]*sum(diag(xy[:,3:4]'*xy[:,3:4]))))

end

function Mprime!{TT}(F::AbstractArray{TT,2},vara::TT,lambda::TT,h::TT,dx::TT,xc::Vector{TT}; n::Int=203)
    
    F[1,1] = one(TT); F[n,n] = one(TT); F[:,2:n-1] = zero(TT)

    ndeltas = max(70,ceil(Int, 10.*sqrt(vara)/dx));

    deltas = collect(-ndeltas:ndeltas) * (5.*sqrt(vara))/ndeltas;
    ps = broadcast(exp, broadcast(/, -broadcast(^, deltas,2), 2.*vara)); ps = ps/sum(ps);

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

                    lp = 1;
                    hp = 2;

                elseif xc[n-1] < s && xc[n] > s

                    lp = n-1;
                    hp = n;

                else

                    hp = ceil(Int, (s-xc[2])/dx) + 2;
                    lp = floor(Int, (s-xc[2])/dx) + 2;

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
