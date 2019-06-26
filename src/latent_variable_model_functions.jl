
const dimz = 7

function compute_deriv(B, pz, py, data::Vector{Dict{Any,Any}}, f_str::String, 
        n::Int)
    
    ll(x) = pulse_input_DDM.compute_LL_test(x, pz, py, data, n, f_str)
    
    ForwardDiff.derivative(ll, B)
            
end

function compute_gradient_2(pz::Dict{}, py::Dict{}, data::Vector{Dict{Any,Any}}, f_str::String, 
        n::Int)
    
    ll(x) = pulse_input_DDM.compute_LL(x, py["generative"], data, n, f_str)
    
    ForwardDiff.gradient(ll, pz["generative"])
        
end

function compute_gradient(pz::Dict{}, py::Dict{}, data::Vector{Dict{Any,Any}}, f_str::String, 
        n::Int)
    
    fit_vec = pulse_input_DDM.combine_latent_and_observation(pz["fit"], py["fit"])
    p_opt, p_const = pulse_input_DDM.split_combine_invmap(pz["final"], py["final"], fit_vec, data[1]["dt"], f_str, pz["lb"], pz["ub"])
    parameter_map_f(x) = pulse_input_DDM.map_split_combine(x, p_const, fit_vec, data[1]["dt"], 
        f_str, py["N"], py["dimy"], pz["lb"], pz["ub"])
    ll(x) = pulse_input_DDM.ll_wrapper(x, data, parameter_map_f, f_str, n)
    
    ForwardDiff.gradient(ll, p_opt)
        
end

function compute_LL_test(B::T, pz::Vector{U}, py::Vector{Vector{Vector{U}}}, data::Vector{Dict{Any,Any}},
        n::Int, f_str::String) where {T,U <: Any}
    
    LL = sum(map((py,data)-> sum(LL_all_trials_test(B, pz, py, data, n, f_str=f_str)), py, data))
            
end

function LL_all_trials_test(B::TT, pz::Vector{UU}, py::Vector{Vector{UU}}, data::Dict, 
        n::Int; f_str::String="softplus") where {TT,UU <: Any}
     
    dt = data["dt"]
    vari = pz[1]
    lambda,vara = pz[3:4]
   
    xc,dx, = bins(B,n)
    
    P = P0(vari,n,dx,xc,dt)
   
    M = zeros(TT,n,n)
    M!(M,vara*dt,lambda,zero(TT),dx,xc,n,dt)  
                            
    output = pmap((L,R,T,nL,nR,SC,λ0) -> LL_single_trial(pz, P, M, dx, xc,
        L, R, T, nL, nR, py, SC, dt, n, λ0, f_str=f_str),
        data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"], 
        data["binned_rightbups"], data["spike_counts"], data["λ0"])   
    
end

function initialize_latent_model(pz::Vector{TT}, n::Int, dt::Float64; 
        L_lapse::UU=0., R_lapse::UU=0.) where {TT,UU}
    
    vari,B,lambda,vara = pz[1:4]                      #break up latent variables
   
    xc,dx,xe = bins(B,n)                              # spatial bin centers, width and edges
    
    P = P0(vari,n,dx,xc,dt; 
        L_lapse=L_lapse, R_lapse=R_lapse)             # make initial latent distribution
   
    M = zeros(TT,n,n)                                 # build empty transition matrix
    M!(M,vara*dt,lambda,zero(TT),dx,xc,n,dt)          # build state transition matrix for no input time bins
    
    return P, M, xc, dx, xe
    
end

function P0(vari::TT, n::Int, dx::VV, xc::Vector{WW}, dt::Float64;
        L_lapse::UU=0., R_lapse::UU=0.) where {TT,UU,VV,WW <: Any}
    
    P = zeros(TT,n)
    P[ceil(Int,n/2)] = one(TT) - (L_lapse + R_lapse)     # make initial delta function
    P[1], P[n] = L_lapse, R_lapse
    M = zeros(WW,n,n)                                    # build empty transition matrix
    M!(M,vari,zero(WW),zero(WW),dx,xc,n,dt)
    P = M * P
    
end

function latent_one_step!(P::Vector{TT},F::Array{TT,2},pz::Vector{WW},t::Int,hereL::Vector{Int}, hereR::Vector{Int},
        La::Vector{YY},Ra::Vector{YY},M::Array{TT,2},
        dx::UU,xc::Vector{VV},n::Int,dt::Float64;backwards::Bool=false) where {TT,UU,VV,WW,YY <: Any}
    
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

function bins(B::TT,n::Int) where {TT}
    
    dx = 2. *B/(n-2);  #bin width
    
    xc = vcat(collect(range(-(B+dx/2.),stop=-dx,length=Int((n-1)/2.))),0.,
        collect(range(dx,stop=(B+dx/2.),length=Int((n-1)/2)))); #centers
    xe = cat(xc[1] - dx/2,xc .+ dx/2, dims=1) #edges
    
    return xc, dx, xe
    
end

myconvert(::Type{T}, x::ForwardDiff.Dual) where {T} = T(x.value)

function M!(F::Array{WW,2},vara::YY,lambda::ZZ,h::Union{TT},dx::UU,xc::Vector{VV}, n::Int, dt::Float64) where {TT,UU,VV,WW,YY,ZZ <: Any}
    
    F[1,1] = one(TT); F[n,n] = one(TT); F[:,2:n-1] = zeros(TT,n,n-2)
       
    #########################################

    #changed 2/17 to keep to less than 1000 bins, haven't checked how that effects returned results
    ndeltas = max(70,ceil(Int, 10. *sqrt(vara)/dx))   
    #ndeltas = 70 + (1000 - 70) * ceil(Int, 0.5*(1+tanh(10. *sqrt(vara)/dx)))
    #ndeltas > 1000 ? (ndeltas = 1000; @warn "using lots of bins!";) : nothing

    #(ndeltas > 1e3 && h == zero(TT)) ? (println(vara); println(dx); println(ndeltas)) : nothing

    #deltas = collect(-ndeltas:ndeltas) * (5.*sqrt(vara))/ndeltas;
    #ps = broadcast(exp, broadcast(/, -broadcast(^, deltas,2), 2.*vara)); ps = ps/sum(ps);
    
    deltaidx = collect(-ndeltas:ndeltas);
    deltas = deltaidx * (5. *sqrt(vara))/ndeltas;
    ps = exp.(-0.5 * (5*deltaidx./ndeltas).^2); ps = ps/sum(ps);
    
    ##########################################
    
    #if !(typeof(vara) == Float64)
    #    sigma2_sbin = myconvert(Float64, vara)
    #    #@warn "this is new!"
    #else
    #    sigma2_sbin = vara
    #end

    #ndeltas = max(70,ceil(Int, 10. *sqrt(sigma2_sbin)/dx))   
    #deltas = collect(-ndeltas:ndeltas) * (5. *sqrt(sigma2_sbin))/ndeltas;
    #ps = exp.(-deltas.^2 / (2. * vara)); ps = ps/sum(ps);
    
    ##########################################    
    
    @inbounds for j = 2:n-1

        abs(lambda) < 1e-150 ? mu = xc[j] + h * dt : mu = exp(lambda*dt)*(xc[j] + h/lambda) - h/lambda
        #testing below, just to use same method for deterministic part as monte carlo data generation
        #tested for noise = 100, high B, lamdbda= -1 and everything came back like other method
        #abs(lambda) < 1e-150 ? mu = xc[j] + h * dt : mu = xc[j]*(1 + dt*lambda) + h * dt
        #lambda == 0. ? mu = xc[j] + h * dt : mu = exp(lambda*dt)*(xc[j] + h/lambda) - h/lambda
        
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

##############################################################################################################

#Testing for faster M matrix, but never finished

#=

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

=#