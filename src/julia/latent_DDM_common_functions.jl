module latent_DDM_common_functions

using StatsBase, Distributions, helpers, DSP, Optim, LineSearches, JLD

const dimz = 7

export diffLR, binLR
export make_adapted_clicks, P0, M!, bins
export sample_latent, construct_inputs!
export decimate, make_observation
export inv_map_pz!, map_pz!, P_M_xc, opt_ll, transition_Pa!
export opt_ll_Newton, gather, inv_gather, group_by_neuron, dimz

function group_by_neuron(data)
    
    trials = Vector{Vector{Int}}()
    SC = Vector{Vector{Vector{Int64}}}()

    map(x->push!(trials,Vector{Int}(0)),1:data["N0"])
    map(x->push!(SC,Vector{Vector{Int}}(0)),1:data["N0"])

    map(y->map(x->push!(trials[x],y),data["N"][y]),1:data["trial0"])
    map(n->map(t->append!(SC[n],data["spike_counts"][t][data["N"][t] .== n]),
        trials[n]),1:data["N0"])
    
    return trials, SC
    
end

inv_gather(p::Vector{TT},fit_vec::Union{BitArray{1},Vector{Bool}}) where TT = p[fit_vec],p[.!fit_vec]

function gather(p_opt::Vector{TT}, p_const::Vector{Float64}, fit_vec::Union{BitArray{1},Vector{Bool}}) where TT
    
    p = Vector{TT}(undef,length(fit_vec))
    p[fit_vec] = p_opt;
    p[.!fit_vec] = p_const;
    
    return p
    
end

function opt_ll(p_opt,ll;g_tol::Float64=1e-12,x_tol::Float64=1e-16,f_tol::Float64=1e-16,iterations::Int=Int(5e3),
        show_trace::Bool=true)
    
    obj = OnceDifferentiable(ll, p_opt; autodiff=:forward)
    m = BFGS(alphaguess = LineSearches.InitialStatic(alpha=1.0,scaled=true), linesearch = BackTracking())
    options = Optim.Options(g_tol = g_tol, x_tol = x_tol, f_tol = f_tol, 
        iterations = iterations, store_trace = true, show_trace = show_trace, 
        extended_trace = false, allow_f_increases = true)
    lbfgsstate = Optim.initial_state(m, options, obj, p_opt)
    
    output = Optim.optimize(obj, p_opt, m, options, lbfgsstate)
    
    return output, lbfgsstate
    
end

opt_ll_Newton(p_opt,ll;g_tol::Float64=1e-12,x_tol::Float64=1e-16,f_tol::Float64=1e-16) = 
        Optim.minimizer(Optim.optimize(OnceDifferentiable(ll, p_opt; autodiff=:forward), 
        p_opt,Newton(alphaguess = LineSearches.InitialStatic(alpha=1.0,scaled=true), 
        linesearch = BackTracking()), Optim.Options(g_tol = g_tol, x_tol = x_tol, f_tol = f_tol, 
        iterations = Int(5e3), store_trace = true, show_trace = true, 
        extended_trace = false, allow_f_increases = true)))

normtanh(x) = 0.5 * (1. + tanh(x))
normatanh(x) = atanh(2. * x - 1.)

function map_pz!(x,dt;map_str::String="exp")
    
    lb = [eps(), 2, -1. /(2*dt), eps(), eps(), eps(), eps()]
    ub = [10., 100, 1. /(2*dt), 800., 40., 2., 10.]
    
    x[3] = lb[3] + (ub[3] - lb[3]) .* normtanh.(x[3])
    
    if map_str == "exp"
        x[[1,2,4,5,6,7]] = lb[[1,2,4,5,6,7]] + exp.(x[[1,2,4,5,6,7]])
    elseif map_str == "tanh"        
        x[[1,2,4,5,6,7]] = lb[[1,2,4,5,6,7]] + (ub[[1,2,4,5,6,7]] - lb[[1,2,4,5,6,7]]) .* normtanh.(x[[1,2,4,5,6,7]])        
    end
        
    return x
    
end

function inv_map_pz!(x,dt;map_str::String="exp")
    
    lb = [eps(), 2, -1. /(2*dt), eps(), eps(), eps(), eps()]
    ub = [10., 100, 1. /(2*dt), 800., 40., 2., 10.]
    
    x[3] = normatanh.((x[3] - lb[3])./(ub[3] - lb[3]))
    
    if map_str == "exp"
        x[[1,2,4,5,6,7]] = log.(x[[1,2,4,5,6,7]] - lb[[1,2,4,5,6,7]])
    elseif map_str == "tanh"
        x[[1,2,4,5,6,7]] = normatanh.((x[[1,2,4,5,6,7]] - lb[[1,2,4,5,6,7]])./(ub[[1,2,4,5,6,7]] - lb[[1,2,4,5,6,7]]))
    
    end
        
    return x
    
end

function P_M_xc(pz::Vector{TT}, n::Int, dt::Float64) where {TT}
    
    #break up latent variables
    vari,B,lambda,vara = pz[1:4]
   
    # spatial bin centers, width and edges
    xc,dx,xe = bins(B,n)
    
    # make initial latent distribution
    P = P0(vari,n,dx,xc,dt);
   
    # build empty transition matrix
    M = zeros(TT,n,n);
    # build state transition matrix for no input time bins
    M!(M,vara*dt,lambda,zero(TT),dx,xc,n,dt)
    
    return P, M, xc, dx, xe
    
end

function transition_Pa!(P::Vector{TT},F::Array{TT,2},pz::Vector{TT},t::Int,hereL::Vector{Int}, hereR::Vector{Int},
        La::Vector{TT},Ra::Vector{TT},M::Array{TT,2},
        dx::TT,xc::Vector{TT},n::Int,dt::Float64;backwards::Bool=false) where {TT}
    
    lambda,vara,vars = pz[3:5]
    
    any(t .== hereL) ? sL = sum(La[t .== hereL]) : sL = zero(TT)
    any(t .== hereR) ? sR = sum(Ra[t .== hereR]) : sR = zero(TT)

    var = vars * (sL + sR);  mu = -sL + sR

    if backwards
        (sL + sR) > zero(TT) ? (M!(F,var+vara*dt,lambda,mu/dt,dx,xc,n,dt); P  = F' * P;) : P = M' * P
    else
        (sL + sR) > zero(TT) ? (M!(F,var+vara*dt,lambda,mu/dt,dx,xc,n,dt); P  = F * P;) : P = M * P
    end
    
    return P, F
    
end

#function make_observation(lambda;dt::Float64=2e-2,noise::String="Poisson",std::Float64=5e0)
    
#    if noise == "Poisson"
#        y = map(lambda->Int(rand(Poisson(lambda*dt))),lambda)
#    elseif noise == "Gaussian"
#        y = map(lambda->rand(Normal(lambda,std)),lambda)
#    end
#            
#end

#function downsample_spiketrain(x;dtMC::Float64=1e-4,dt::Float64=2e-2)
#    
#    nbins = Int(dt/dtMC)    
#    x = vcat(x, mean(x[end-rem(length(x),nbins)+1:end]) * ones(eltype(x),mod(nbins - rem(length(x),nbins),nbins)))
#    x = squeeze(mean(reshape(x,nbins,:),1),1)  
#    
#    return x
#    
#end

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

function construct_inputs!(data::Dict,num_reps::Int)
    
    dt = data["dt"]
    binnedT = ceil.(Int,data["T"]/dt);

    data["nT"] = binnedT
    data["binned_leftbups"] = map((x,y)->vec(qfind(0.:dt:x*dt,y)),binnedT,data["leftbups"])
    data["binned_rightbups"] = map((x,y)->vec(qfind(0.:dt:x*dt,y)),binnedT,data["rightbups"])
    
    #use repmat to make as any copys as needed
    data["nT"] = repmat(data["nT"],num_reps)
    data["binned_leftbups"] = repmat(data["binned_leftbups"],num_reps)
    data["binned_rightbups"] = repmat(data["binned_rightbups"],num_reps)
    data["T"] = repmat(data["T"],num_reps)
    data["leftbups"] = repmat(data["leftbups"],num_reps)
    data["rightbups"] = repmat(data["rightbups"],num_reps)
    data["trial0"] = data["trial0"] * num_reps;
    
    if haskey(data,"N")
        data["N"] = repmat(data["N"],num_reps)   
    end
    
    return data
    
end

function sample_clicks(ntrials::Int,dt::Float64)
    
    data = Dict();

    output = map(generate_stimulus,1:ntrials);

    data["leftbups"] = map(i->output[i][3],1:ntrials);
    data["rightbups"] = map(i->output[i][2],1:ntrials);
    data["T"] = map(i->output[i][1],1:ntrials);
    data["dt"] = dt;
    data["trial0"] = ntrials;

    #bin the clicks
    data["nT"] = ceil.(Int,data["T"]/dt);
    data["binned_leftbups"] = map((x,y)->vec(qfind(0.:dt:x*dt,y)),data["nT"],data["leftbups"])
    data["binned_rightbups"] = map((x,y)->vec(qfind(0.:dt:x*dt,y)),data["nT"],data["rightbups"])
    
    return data
    
end

function generate_stimulus(rng;tmin::Float64=0.2,tmax::Float64=1.0,clicktot::Int=40)
    
    srand(rng)

    T = tmin + (tmax-tmin)*rand()

    ratetot = clicktot/T
    Rbar = ratetot*rand()
    Lbar = ratetot - Rbar

    R = cumsum(rand(Exponential(1/Rbar),clicktot))
    L = cumsum(rand(Exponential(1/Lbar),clicktot))
    R = vcat(0,R[R .<= T])
    L = vcat(0,L[L .<= T])
    
    return T,R,L
    
end

function sample_latent(T::Float64,L::Vector{Float64},R::Vector{Float64},
        pz::Vector{Float64};dt::Float64=1e-4)
    
    vari, B, lambda, vara, vars, phi, tau_phi = pz;
    
    nT = Int(ceil.(T/dt)); # number of timesteps

    La, Ra = make_adapted_clicks(pz,L,R)
    t = 0.:dt:nT*dt-dt; 
    hereL = vec(qfind(t,L))
    hereR = vec(qfind(t,R))

    A = Vector{Float64}(nT)
    a = sqrt(vari)*randn()

    for t = 1:nT

        #inputs
        any(t .== hereL) ? sL = sum(La[t .== hereL]) : sL = zero(phi)
        any(t .== hereR) ? sR = sum(Ra[t .== hereR]) : sR = zero(phi)
        var = vars * (sL + sR)  
        mu = -sL + sR
        (sL + sR) > 0. ? a += mu + sqrt(var) * randn() : nothing

        #drift and diffuse
        a += (dt*lambda) * a + sqrt(vara * dt) * randn();

        abs(a) > B ? (a = B * sign(a); A[t:nT] = a; break) : A[t] = a

    end               
    
    return A
    
end

function bins(B::TT,n::Int) where {TT}
    
    dx = 2. *B/(n-2);  #bin width
    
    xc = vcat(collect(range(-(B+dx/2.),stop=-dx,length=Int((n-1)/2.))),0.,collect(range(dx,stop=(B+dx/2.),length=Int((n-1)/2)))); #centers
    xe = cat(xc[1] - dx/2,xc .+ dx/2, dims=1) #edges
    
    return xc, dx, xe
    
end

function P0(vari::TT,n::Int,dx::TT,xc::Vector{TT},dt::Float64) where {TT}
    
    # make initial delta function
    P = zeros(TT,n); 
    P[ceil(Int,n/2)] = one(TT); 
    # build empty transition matrix
    M = zeros(TT,n,n);
    M!(M,vari,zero(TT),zero(TT),dx,xc,n,dt); 
    P = M * P
    
end

function M!(F::Array{TT,2},vara::TT,lambda::TT,h::Union{TT,Float64},dx::TT,xc::Vector{TT}, n::Int, dt::Float64) where {TT}
    
    F[1,1] = one(TT); F[n,n] = one(TT); F[:,2:n-1] = zeros(TT,n,n-2)

    ndeltas = max(70,ceil(Int, 10. *sqrt(vara)/dx));

    (ndeltas > 1e3 && h == zero(TT)) ? (println(vara); println(dx); println(ndeltas)) : nothing

    #deltas = collect(-ndeltas:ndeltas) * (5.*sqrt(vara))/ndeltas;
    #ps = broadcast(exp, broadcast(/, -broadcast(^, deltas,2), 2.*vara)); ps = ps/sum(ps);
    
    deltaidx = collect(-ndeltas:ndeltas);
    deltas = deltaidx * (5. *sqrt(vara))/ndeltas;
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

muf(x,lambda,h,dt) = abs(lambda) < 1e-150 ? mu = x + h * dt : mu = exp(lambda*dt)*(x + h/lambda) - h/lambda

function newM!(F::Array{TT,2},vara::TT,lambda::TT,h::Union{TT,Float64},dx::TT,xc::Vector{TT}; n::Int=203, dt::Float64=2e-2) where {TT}
    
    #F[1,1] = one(TT); F[n,n] = one(TT); F[:,2:n-1] = zero(TT)
    F[1,1], F[n,n], F[:,2:n-1] = one(TT), one(TT), zero(TT)

    #ndeltas = max(70,ceil(Int, 10.*sqrt(vara)/dx));

    #(ndeltas > 1e3 && h == zero(TT)) ? (println(vara); println(dx); println(ndeltas)) : nothing

    #deltas = collect(-ndeltas:ndeltas) * (5.*sqrt(vara))/ndeltas;
    #ps = broadcast(exp, broadcast(/, -broadcast(^, deltas,2), 2.*vara)); ps = ps/sum(ps);
    
    #deltaidx = collect(-ndeltas:ndeltas);
    #deltas = deltaidx * (5.*sqrt(vara))/ndeltas;
    #ps = exp.(-0.5 * (5*deltaidx./ndeltas).^2); ps = ps/sum(ps);
    
    ndeltas = max(70,ceil(Int, 10. *sqrt(vara)/dx));   
    deltas = 5. *sqrt(vara) * linspace(-1.,1.,2*ndeltas+1)
    ps = pdf.(Normal(0,sqrt(vara)),deltas); ps /= sum(ps)
    
    @inbounds for j = 2:n-1

        #mu = muf(xc[j],lambda,h,dt)
        #abs(lambda) < 1e-150 ? mu = xc[j] + h * dt : mu = exp(lambda*dt)*(xc[j] + h/lambda) - h/lambda
        
        #now we're going to look over all the slices of the gaussian
        @inbounds for k = 1:2*ndeltas+1

            s = muf(xc[j],lambda,h,dt) + deltas[k]

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

function make_adapted_clicks(pz::Vector{TT}, L::Vector{Float64}, R::Vector{Float64}) where {TT}
    
    #break up parameters
    phi,tau_phi = pz[6:7]

    La, Ra = ones(TT,length(L)), ones(TT,length(R))

    # magnitude of stereo clicks set to zero
    # I removed these lines on 8/8/18, because I'm not exactly sure why they are here (from Bing's original model)
    # and the cause the state to adapt even when phi = 1., which I'd like to spend time fitting simpler models to
    # check slack discussion with adrian and alex
    
    #if !isempty(L) && !isempty(R) && abs(L[1]-R[1]) < eps()
    #    La[1], Ra[1] = eps(), eps()
    #end

    (length(L) > 1 && phi != 1.) ? (ici_L = diff(L); adapt_clicks!(La,phi,tau_phi,ici_L)) : nothing
    (length(R) > 1 && phi != 1.) ? (ici_R = diff(R); adapt_clicks!(Ra,phi,tau_phi,ici_R)) : nothing
    
    return La, Ra

end

function adapt_clicks!(Ca::Vector{TT}, phi::TT, tau_phi::TT, ici::Vector{Float64}) where {TT}
    
    for i = 1:length(ici)
        #Change this on 11/4 because was getting NaNs when tau_phi = 0.
        arg = abs(1. - Ca[i]*phi)
        arg > 1e-150 ? Ca[i+1] = 1. - exp((-ici[i] + tau_phi*log(arg))/tau_phi) : nothing
        #changed back on 11/5 because realized problem was really with weird gen parameters
        #and checked that LL was same for either way when using better gerative parameters
        #arg = tau_phi*log(abs(1. - Ca[i]*phi))
        #arg > 1e-150 ? Ca[i+1] = 1. - exp((-ici[i] + arg)/tau_phi) : Ca[i+1] = one(TT)
        #arg = (-ici[i] + tau_phi*log(abs(1. - Ca[i]*phi)))/tau_phi
        #abs(arg) > 1e-150 ? Ca[i+1] = 1. - exp(arg) : Ca[i+1] = one(TT)
    end
    
end

#should modify this to return the differnece only, and then allow filtering afterwards
function diffLR(nT,L,R,dt)
    
    L,R = binLR(nT,L,R,dt)   
    cumsum(-L + R)
    
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

function my_callback(os)

    #so_far = time() - start_time
    #println(" * Time so far:     ", so_far)
  
    #history = Array{Float64,2}(sum(fit_vec),0)
    #history_gx = Array{Float64,2}(sum(fit_vec),0)
    #for i = 1:length(os)
    #    ptemp = group_params(os[i].metadata["x"], p_const, fit_vec)
    #    ptemp = map_func!(ptemp,model_type,"tanh",N=N)
    #    ptemp_opt, = break_params(ptemp, fit_vec)       
    #    history = cat(2,history,ptemp_opt)
    #    history_gx = cat(2,history_gx,os[i].metadata["g(x)"])
    #end
    print(os[1]["x"])
    #save(ENV["HOME"]*"/spike-data_latent-accum"*"/history.jld", "os", os)
    #print(path)

    return false

end

end
