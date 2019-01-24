module module_DDM

const dt = 2e-2
const n = 203
const nstdc = 6
const settle = true
const dim_z = 8
const dim_d = 1

import ForwardDiff
import Base.convert
using StatsBase

export LL_all_trials
export sample_model
export LL_single_trial
export qfind
export do_conv!
export my_conv
export process_rawdata!
export process_rawdata_single_session!
export my_expm
export init_xy
export convert_data!
export LL_all_trials_unc

convert(::Type{Float64}, x::ForwardDiff.Dual) = Float64(x.value)
convert(::Type{Int},x::ForwardDiff.Dual) = Int(x.value)

function convert_data!(data)

    data["T"] = convert(Array{Float64,2},data["T"])
    data["nT"] = convert(Array{Int,2},data["nT"])
    data["Ntotal"] = convert(Int,data["Ntotal"][1])
    data["here_L"] = data["hereL"]
    data["here_R"] = data["hereR"]
    for i = 1:length(data["hereL"])

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
        
        if length(data["here_L"][i]) > 1
            data["here_L"][i] = vec(convert(Array{Int},data["here_L"][i]))
        else
            data["here_L"][i] = convert(Int,data["here_L"][i])
        end

        if length(data["here_R"][i]) > 1
            data["here_R"][i] = vec(convert(Array{Int},data["here_R"][i]))
        else
            data["here_R"][i] = convert(Int,data["here_R"][i])
        end
    end

end

function init_xy(rawdata)

    N = size(rawdata["St"][1],2)
    temp = [Array{Float64}(0,2) for idx in 1:N]

    for i = 1:length(rawdata["T"])
        t = 0.:dt:rawdata["nT"][i]*dt

        if length(rawdata["leftbups"][i]) == 1
            L = reshape(collect(rawdata["leftbups"][i]),1)
        else
            L = vec(rawdata["leftbups"][i])
        end

        if length(rawdata["rightbups"][i]) == 1
            R = reshape(collect(rawdata["rightbups"][i]),1)
        else
            R = vec(rawdata["rightbups"][i])
        end

        resultsL = fit(Histogram,L,t,closed=:left)
        resultsR = fit(Histogram,R,t,closed=:left)

        diffLR = cumsum(-resultsL.weights + resultsR.weights)

        if  Float64(rawdata["pokedR"][i]) == 0.5 * (sign(diffLR[end]) + 1.)
            for j = 1:length(rawdata["N"][i])
                temp[rawdata["N"][i][j]] = vcat(temp[rawdata["N"][i][j]],hcat(rawdata["spike_counts"][i][:,j],diffLR))
            end
        end

    end

    B = maximum(temp[1][:,2])
    xy = Array{Float64}(4,N)

    for i = 1:N

        b = \(hcat(temp[i][:,2],ones(temp[i][:,2])),temp[i][:,1])

        if b[1] > 0.

            xy[1,i] = b[1] * -B + b[2]
            xy[2,i] = b[1] * B + b[2] - xy[1,i]

        else

            xy[1,i] = b[1] * B + b[2]
            xy[2,i] = b[1] * -B + b[2] - xy[1,i]

        end

        xy[3,i] = sign(b[1]) * 1/B
        xy[4,i] = 0.
    end 

    return xy

end

function my_expm(A)
    
    e = ceil(Int,log2(norm(A,Inf)))
    s = maximum([0,e+1])
    A /= 2^s
    X = copy(A)
    c = 1./2.
    E = eye(A) + c*A
    D = eye(A) - c*A

    q = 6
    p = true

    for k = 2:q
        c *= (q - k + 1)/ (k * (2*q-k+1))
        X = A * X
        cX = c * X
        E += cX
        p ? D += cX : D -= cX
        p = ~p
    end

    E = D\E

    for k = 1:s
        E = E * E
    end

    return E

end

function process_rawdata_single_session!(rawdata,use)

    ntrials = length(rawdata["T"])
    rawdata["nT"] = ceil.(Int,rawdata["T"]/dt)
    rawdata["spike_counts"] = Array{Array{Int}}(ntrials)
    rawdata["N"] = Array{Array{Int}}(ntrials)
    rawdata["here_L"] = Array{Array{Int}}(ntrials)
    rawdata["here_R"] = Array{Array{Int}}(ntrials)
    rawdata["here_LR"] = Array{Array{Int}}(ntrials)

    for i = 1:ntrials
        t = 0.:dt:rawdata["nT"][i]*dt;
        rawdata["here_L"][i] = vec(qfind(t,rawdata["leftbups"][i]))
        rawdata["here_R"][i] = vec(qfind(t,rawdata["rightbups"][i]))
        rawdata["here_LR"][i] = sort(unique([rawdata["here_L"][i];rawdata["here_R"][i]]))
    end

    if use["spikes"]

        N = 0
        Nid = Array{Int}(0)
        for i = 1:size(rawdata["St"][1],2)
            if size(rawdata["St"][1][i],2) > 0
                N = N + 1;
                Nid = vcat(Nid,i)
            end
        end

        rawdata["Ntotal"] = N

        for i = 1:ntrials

            t = 0.:dt:rawdata["nT"][i]*dt;
            temp = Array{Int}(rawdata["nT"][i],0)
            tempN = Array{Int}(0);

            for j = 1:N

                if size(rawdata["St"][i][Nid[j]],2) > 0

                    if length(rawdata["St"][i][Nid[j]]) == 1
                        temp2 = reshape(collect(rawdata["St"][i][Nid[j]]),1)
                    else
                        temp2 = vec(rawdata["St"][i][Nid[j]])
                    end

                    results = fit(Histogram,temp2,t,closed=:left)
                    temp = hcat(temp,results.weights)
                    tempN = vcat(tempN,j)

                end
            end

            rawdata["spike_counts"][i] = temp
            rawdata["N"][i] = tempN
        end

    end

    if use["choice"] && ~use["spikes"]
        for i = 1:ntrials
            rawdata["N"][i] = [];
            rawdata["spike_counts"][i] = Array{Int}(0,2)
        end
    end

end

function process_rawdata!(rawdata,use)

    ntrials = length(rawdata["T"])
    rawdata["nT"] = ceil.(Int,rawdata["T"]/dt)
    rawdata["spike_counts"] = Array{Array{Int}}(ntrials)
    rawdata["N"] = Array{Array{Int}}(ntrials)
    rawdata["here_L"] = Array{Array{Int}}(ntrials)
    rawdata["here_R"] = Array{Array{Int}}(ntrials)
    rawdata["here_LR"] = Array{Array{Int}}(ntrials)

    for i = 1:ntrials
        t = 0.:dt:rawdata["nT"][i]*dt;
        rawdata["here_L"][i] = vec(qfind(t,rawdata["leftbups"][i]))
        rawdata["here_R"][i] = vec(qfind(t,rawdata["rightbups"][i]))
        rawdata["here_LR"][i] = sort(unique([rawdata["here_L"][i];rawdata["here_R"][i]]))
    end

    if use["spikes"]

        N = size(rawdata["St"][1],2)
        rawdata["Ntotal"] = N

        for i = 1:ntrials

            t = 0.:dt:rawdata["nT"][i]*dt;
            temp = Array{Int}(rawdata["nT"][i],0)
            tempN = Array{Int}(0);

            for j = 1:N

                if size(rawdata["St"][i][j],2) > 0

                    if length(rawdata["St"][i][j]) == 1
                        temp2 = reshape(collect(rawdata["St"][i][j]),1)
                    else
                        temp2 = vec(rawdata["St"][i][j])
                    end

                    results = fit(Histogram,temp2,t,closed=:left)
                    temp = hcat(temp,results.weights)
                    tempN = vcat(tempN,j)

                end
            end

            rawdata["spike_counts"][i] = temp
            rawdata["N"][i] = tempN
        end

    end

    if use["choice"] && ~use["spikes"]
        for i = 1:ntrials
            rawdata["N"][i] = [];
            rawdata["spike_counts"][i] = Array{Int}(0,2)
        end
    end

end

function make_adapted_clicks(leftbups, rightbups, phi, tau_phi)

    L = ones(typeof(phi),size(leftbups));
    R = ones(typeof(phi),size(rightbups));

    # magnitude of stereo clicks set to zero
    if ~isempty(leftbups) && ~isempty(rightbups) && abs(leftbups[1]-rightbups[1]) < eps()
        L[1] = eps()
        R[1] = eps()
    end

    # if there's appreciable same-side adaptation
    #if abs(phi - 1) > eps() 

        if length(leftbups) <= 1
            ici_l = [];
        else
            ici_L = (leftbups[2:end]  - leftbups[1:end-1])';
        end

        if length(rightbups) <= 1
            ici_R = [];
        else
            ici_R = (rightbups[2:end]  - rightbups[1:end-1])';
        end

        for i = 2:length(leftbups),
            last_L = tau_phi*log(abs(1-L[i-1]*phi));
            L[i] = 1 - exp((-ici_L[i-1] + last_L)/tau_phi);
        end;

        for i = 2:length(rightbups),
            last_R = tau_phi*log(abs(1-R[i-1]*phi));
            R[i] = 1 - exp((-ici_R[i-1] + last_R)/tau_phi);
        end;

        L = real(L);
        R = real(R);

    #end

    return L, R

end

function my_conv{TT}(u::Vector{TT},v::Vector{TT})

    nu = length(u)
    nv = length(v)
    w = zeros(TT,nu+nv-1)

    @inbounds for i = 1:nu
        for j = 1:nv
            w[i+j-1] += u[i]*v[j]
        end
    end

    return w

end

function Mprime!{TT}(F::AbstractArray{TT,2},vara::TT,lambda::TT,h::TT,dx::TT,xc::Vector{TT})
    
    F[1,1] = 1.; F[n,n] = 1.
    @inbounds for j = 2:n-1;  for k = 1:n;  F[k,j] = 0.; end; end

    ndeltas = max(70,ceil(Int, 10.*sqrt(vara)/dx));

    deltas = collect(-ndeltas:ndeltas) * (5.*sqrt(vara))/ndeltas;
    ps = broadcast(exp, broadcast(/, -broadcast(^, deltas,2), 2.*vara)); ps = ps/sum(ps);

    @inbounds for j = 2:n-1

        abs(lambda) < 1e-10 ? mu = xc[j] + h * dt : mu = exp(lambda*dt)*(xc[j] + h/lambda) - h/lambda

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

function do_conv!{TT}(mu::TT,var::TT,P::Vector{TT},dx::TT)
    
    iL = minimum([0;floor(Int,(mu-nstdc*sqrt(var))/dx)]); # left edge of support (in # bins)
    iR =  maximum([0;ceil(Int,(mu+nstdc*sqrt(var))/dx)]); # right edge of support (in # bins)
    x = (iL:iR)*dx
    pcnv = dx/sqrt(2.*pi*var) * exp.(-(x-mu).^2/(2*var));
    pcnv = pcnv/sum(pcnv) #normalize

    Pfull = my_conv(P[2:n-1],pcnv)
    P[1] += sum(Pfull[1:-iL])
    P[2:n-1] = Pfull[-iL+1:end-iR]
    P[n] += sum(Pfull[end-iR+1:end])  # mass in last bin

end

function do_M{TT}(vara::TT,lambda::TT,dx::TT,xc::Array{TT,1})

    vara /= dx^2*2.   # scale factor for diffusion
    lambda /= dx*2.   # scale factor for drift

    vara *= Tridiagonal(vcat(zero(TT), ones(TT,n-2)), -2. * vcat(zero(TT),ones(TT,n-2),zero(TT)), vcat(ones(TT,n-2),zero(TT)));  # diffusion matrix
    lambda *= Tridiagonal(-vcat(zero(TT),ones(TT,n-2)), zeros(xc), vcat(ones(TT,n-2),zero(TT)));
    lambda *= diagm(xc);
    vara -= lambda;
    vara *= dt;

    M = my_expm(vara)
    return M

end

function LL_single_trial{TT}(x::Array{TT,1}, P::Array{TT,1}, M::Array{TT,2}, dx::TT, xc::Array{TT,1}, T::Int, L, R, here_L::Union{Int,Array{Int,1}}, here_R::Union{Int,Array{Int,1}},
                             use::Dict, nbinsL::Union{Int,TT}, Sfrac::Union{Float64,TT}, pokedR::Bool, lambda::Union{Array{TT,2},Array{TT,1}}, spike_counts::Union{Array{Int,1},Array{Int,2}})

    vars = x[1];  phi = x[2];  tau_phi = x[3]

    La, Ra = make_adapted_clicks(L,R,phi,tau_phi)

    notpoked = convert(TT,~pokedR); poked = convert(TT,pokedR)
    use["choice"] ? Pd = vcat(notpoked * ones(nbinsL), notpoked * Sfrac + poked * (one(Sfrac) - Sfrac), poked * ones(n - (nbinsL + 1))) : nothing 
    use["spikes"] ? Py = exp.(broadcast(-, broadcast(-, spike_counts *  log.(lambda'*dt), sum(lambda,2)' * dt), sum(lgamma.(spike_counts + 1),2)))' : nothing
    LL = zero(TT);

    @inbounds for t = 1:T
        
        any(t .== here_L) ? sL = sum(La[t .== here_L]) : sL = zero(phi)
        any(t .== here_R) ? sR = sum(Ra[t .== here_R]) : sR = zero(phi)

        var = vars * (sL + sR);  mu = -sL + sR

        if var > zero(vars)

            ((var < dx^2) | settle) ? (isdefined(:F) ||  (F = zeros(M));  Mprime!(F,var,zero(TT),mu/dt,dx,xc); P  = F * P) : do_conv!(mu,var,P,dx);

        end

        P = M * P
        
        use["spikes"] && (P .*= Py[:,t])
        use["choice"] && t == T && (P .*=  Pd)

        LL += log(abs(sum(P) + eps()))
        P /= (sum(P) + eps()) 

    end

    return LL
end

function LL_all_trials{TT}(xf::Vector{TT}, rawdata::Dict, use::Dict, x0::Vector{Float64}, fit_vec::Array{Bool,1})

    x = Array{TT}(length(fit_vec))
    x[fit_vec] = xf;
    x[.!fit_vec] = x0[.!fit_vec];

    vari = x[1]; inatt = x[2];  B = x[3]; lambda = x[4];  vara = x[5]; 
   
    # binning
    dx = 2.*B/(n-2);  #bin width
    xc = vcat(collect(linspace(-(B+dx/2.),-dx,(n-1)/2.)),0.,collect(linspace(dx,(B+dx/2.),(n-1)/2)));

    # build state transition matrix
    (((abs(lambda) * B)/vara > one(TT)/dx) | settle) ? (M = zeros(TT,n,n);  Mprime!(M,vara*dt,lambda,zero(TT),dx,xc)) :  M = do_M(vara,lambda,dx,xc)

    # make initial delta function
    P = zeros(xc); P[[1,n]] = inatt/2.; P[ceil(Int,n/2)] = one(TT) - inatt; 
    # Convolve initial delta with vari
    ((vari < dx^2) | settle) ? (M0 = zeros(M);  Mprime!(M0,vari,zero(TT),zero(TT),dx,xc); P = M0 * P) : do_conv!(zero(TT),vari,P,dx)

    if use["choice"]
        bias = x[9]
        nbinsL = ceil(Int,(B+bias)/dx)
        Sfrac = one(dx)/dx * (bias - (-(B+dx)+nbinsL*dx))
    else
        Sfrac = 1.; nbinsL = 1
    end

    if use["spikes"]
        #N = size(rawdata["St"][1],2)
        N = rawdata["Ntotal"]
        use["choice"] ? (a = x[(1:N)+dim_z+dim_d]; b = x[(1:N)+dim_z+dim_d+N]; c = x[(1:N)+dim_z+dim_d+2*N]; d = x[(1:N)+dim_z+dim_d+3*N]) :
            (a = x[(1:N)+dim_z]; b = x[(1:N)+dim_z+N]; c = x[(1:N)+dim_z+2*N]; d = x[(1:N)+dim_z+3*N])
        lambda = broadcast(+, a', broadcast(/, b', (1 + broadcast(exp, broadcast(+, broadcast(*, -c', xc), d')))))
    else
        lambda = Array{TT}(0,2);
    end

    LL =  @parallel (+) for i = 1:length(rawdata["T"])
        LL_single_trial(x[6:8],copy(P),M,dx,xc,rawdata["nT"][i],rawdata["leftbups"][i],rawdata["rightbups"][i],rawdata["here_L"][i],rawdata["here_R"][i],
                        use,nbinsL,Sfrac,Bool(rawdata["pokedR"][i]),lambda[:,rawdata["N"][i]],rawdata["spike_counts"][i])
    end

    return -LL

end

function LL_all_trials_unc{TT}(xf::Vector{TT}, rawdata::Dict, use::Dict, x0::Vector{Float64}, fit_vec::Array{Bool,1},lb::Vector{Float64},ub::Vector{Float64})

    x = Array{TT}(length(fit_vec))
    x[fit_vec] = xf;
    x[.!fit_vec] = x0[.!fit_vec];

    x = lb + (ub-lb) .* 0.5 .* (1 + tanh.(x));

    vari = x[1]; inatt = x[2];  B = x[3]; lambda = x[4];  vara = x[5]; 
   
    # binning
    dx = 2.*B/(n-2);  #bin width
    xc = vcat(collect(linspace(-(B+dx/2.),-dx,(n-1)/2.)),0.,collect(linspace(dx,(B+dx/2.),(n-1)/2)));

    # build state transition matrix
    (((abs(lambda) * B)/vara > one(TT)/dx) | settle) ? (M = zeros(TT,n,n);  Mprime!(M,vara*dt,lambda,zero(TT),dx,xc)) :  M = do_M(vara,lambda,dx,xc)

    # make initial delta function
    P = zeros(xc); P[[1,n]] = inatt/2.; P[ceil(Int,n/2)] = one(TT) - inatt; 
    # Convolve initial delta with vari
    ((vari < dx^2) | settle) ? (M0 = zeros(M);  Mprime!(M0,vari,zero(TT),zero(TT),dx,xc); P = M0 * P) : do_conv!(zero(TT),vari,P,dx)

    if use["choice"]
        bias = x[9]
        nbinsL = ceil(Int,(B+bias)/dx)
        Sfrac = one(dx)/dx * (bias - (-(B+dx)+nbinsL*dx))
    else
        Sfrac = 1.; nbinsL = 1
    end

    if use["spikes"]
        #N = size(rawdata["St"][1],2)
        N = rawdata["Ntotal"]
        use["choice"] ? (a = x[(1:N)+dim_z+dim_d]; b = x[(1:N)+dim_z+dim_d+N]; c = x[(1:N)+dim_z+dim_d+2*N]; d = x[(1:N)+dim_z+dim_d+3*N]) :
            (a = x[(1:N)+dim_z]; b = x[(1:N)+dim_z+N]; c = x[(1:N)+dim_z+2*N]; d = x[(1:N)+dim_z+3*N])
        lambda = broadcast(+, a', broadcast(/, b', (1 + broadcast(exp, broadcast(+, broadcast(*, -c', xc), d')))))
    else
        lambda = Array{TT}(0,2);
    end

    LL =  @parallel (+) for i = 1:length(rawdata["T"])
        LL_single_trial(x[6:8],copy(P),M,dx,xc,rawdata["nT"][i],rawdata["leftbups"][i],rawdata["rightbups"][i],rawdata["here_L"][i],rawdata["here_R"][i],
                        use,nbinsL,Sfrac,Bool(rawdata["pokedR"][i]),lambda[:,rawdata["N"][i]],rawdata["spike_counts"][i])
    end

    return -LL

end

function sample_model(x, rawdata; dt::Float64=1e-4)

    #latent variable model
    vari = x[1]; #initial variance
    inatt = x[2]; #lapse rate (choose a random choice)
    B = x[3]; #bound height
    lambda = x[4]; #1/tau, where tau is the decay timescale
    vara = x[5]; #diffusion variance
    vars = x[6]; #stimulus variance
    phi = x[7]; #click adaptation magnitude
    tau_phi = x[8]; #adaptation decay timescale
    bias = x[9];

    ntrials = length(rawdata["T"])
    choice = SharedArray{typeof(B)}(ntrials);

    @inbounds @sync @parallel for i = 1:ntrials
        choice[i] = sample_single_trial(rawdata["T"][i], rawdata["leftbups"][i], rawdata["rightbups"][i],dt,x,i)
    end
    #choice = pmap(x,y,z->sample_single_trial(rawdata["T"],rawdata["leftbups"],rawdata["rightbups"],dt,x),rawdata["T"],rawdata["leftbups"],rawdata["rightbups"]);

    return choice

end

function sample_single_trial(T,L,R,dt,x,i)

    srand(i);
    #latent variable model
    vari = x[1]; #initial variance
    inatt = x[2]; #lapse rate (choose a random choice)
    B = x[3]; #bound height
    lambda = x[4]; #1/tau, where tau is the decay timescale
    vara = x[5]; #diffusion variance
    vars = x[6]; #stimulus variance
    phi = x[7]; #click adaptation magnitude
    tau_phi = x[8]; #adaptation decay timescale
    bias = x[9];

    nT = Int(ceil(T/dt));	
    La,Ra = make_adapted_clicks(L,R,phi,tau_phi)
    timevec = 0:dt:nT*dt;

    a = sqrt(vari) * randn(1); #initialize particles

    if any(rand(1) .< inatt)

        a = B * sign.(randn(1));
        A = a;

    else

        A = zeros(Float64,length(timevec)-1);

        for i = 1:nT

            if any(a .> -B) .& any(a .< B)

                Lvec = (L .>= timevec[i]) .& (L .< timevec[i+1]);
                Rvec = (R .>= timevec[i]) .& (R .< timevec[i+1]);

                if any(Lvec) | any(Rvec)

                   if any(Lvec)
                       sL = sum(La[Lvec]);
                    else
                        sL = 0.0;
                    end

                    if any(Rvec)
                        sR = sum(Ra[Rvec]);
                    else
                        sR = 0.0;
                    end

                    var = vars * (sL + sR);
                    mu = -sL + sR;

                    a = a + (dt*lambda) * a + mu + sqrt(vara * dt) * randn(1) + sqrt(var) * randn(1);

                else

                    a = a + (dt*lambda) * a + sqrt(vara * dt) * randn(1);

                end

            end

            A[i] = a[1];

        end

    end

    choice = any(a .> bias);

    return choice

end

end
