function construct_inputs!(data::Dict,num_reps::Int)
    
    dt = data["dt"]
    binnedT = ceil.(Int,data["T"]/dt);

    data["nT"] = binnedT
    data["binned_leftbups"] = map((x,y)->vec(qfind(0.:dt:x*dt,y)),binnedT,data["leftbups"])
    data["binned_rightbups"] = map((x,y)->vec(qfind(0.:dt:x*dt,y)),binnedT,data["rightbups"])
    
    #use repmat to make as any copys as needed
    data["nT"] = repeat(data["nT"],inner=num_reps)
    data["binned_leftbups"] = repeat(data["binned_leftbups"],inner=num_reps)
    data["binned_rightbups"] = repeat(data["binned_rightbups"],inner=num_reps)
    data["T"] = repeat(data["T"],inner=num_reps)
    data["leftbups"] = repeat(data["leftbups"],inner=num_reps)
    data["rightbups"] = repeat(data["rightbups"],inner=num_reps)
    data["trial0"] = data["trial0"] * num_reps;
    
    if haskey(data,"N")
        data["N"] = repeat(data["N"],inner=num_reps)   
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
    
    Random.seed!(rng)

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

    A = Vector{Float64}(undef,nT)
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

        abs(a) > B ? (a = B * sign(a); A[t:nT] .= a; break) : A[t] = a

    end               
    
    return A
    
end

#################################### Choice observation model #################################

function sampled_dataset!(data::Dict, pz::Vector{Float64}, pd::Vector{Float64}; 
        dtMC::Float64=1e-4, num_reps::Int=1, rng::Int = 1)
        
    construct_inputs!(data,num_reps)
    
    Random.seed!(rng)
    data["pokedR"] = pmap((T,leftbups,rightbups,rng) -> sample_model(T,leftbups,rightbups,pz,pd,rng=rng),
        data["T"],data["leftbups"],data["rightbups"], shuffle(1:length(data["T"])));
            
    return data
    
end

function sample_model(T::Float64,L::Vector{Float64},R::Vector{Float64},
        pz::Vector{Float64},pd::Vector{Float64};dtMC::Float64=1e-4,rng::Int=1)
    
    Random.seed!(rng)
    
    A = sample_latent(T,L,R,pz;dt=dtMC)
    
    bias,lapse = pd[1],pd[2]
            
    rand() > lapse ? choice = A[end] >= bias : choice = Bool(round(rand()))
    
end

#################################### Poisson neural observation model #########################

function sampled_dataset!(data::Dict, pz::Vector{Float64}, py::Vector{Vector{Float64}},
        dt::Float64; f_str::String="softplus", dtMC::Float64=1e-4, num_reps::Int=1, rng::Int=1)

    construct_inputs!(data,num_reps)
    
    Random.seed!(rng)
    data["spike_counts"] = pmap((T,L,R,N,rng) -> sample_model(pz,py,T,L,R,N,dt;
            f_str=f_str, rng=rng), data["T"], data["leftbups"], data["rightbups"],
            data["N"], shuffle(1:length(data["T"])));        
    
    return data
    
end

function sample_model(pz, py, T::Float64, L::Vector{Float64}, R::Vector{Float64},
         N::Vector{Int}, dt::Float64; f_str::String="softplus", dtMC::Float64=1e-4, rng::Int=1,
         λ0::Vector{Float64}=Vector{Float64}())
    
    Random.seed!(rng)
    
    A = decimate(sample_latent(T,L,R,pz;dt=dtMC), Int(dt/dtMC))  
    
    Y = map(py-> poisson_noise.(fy22(py, A, λ0, f_str=f_str), dt), py[N])       
    
end

poisson_noise(lambda,dt) = Int(rand(Poisson(lambda*dt)))