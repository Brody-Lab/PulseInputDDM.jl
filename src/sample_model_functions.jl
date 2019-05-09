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

function sample_clicks(ntrials::Int; rng::Int=1)
    
    Random.seed!(rng)
    
    data = Dict()

    output = map(generate_stimulus,1:ntrials)

    data["leftbups"] = map(i->output[i][3],1:ntrials)
    data["rightbups"] = map(i->output[i][2],1:ntrials)
    data["T"] = map(i->output[i][1],1:ntrials)
    data["ntrials"] = ntrials
    
    return data
    
end

function generate_stimulus(i::Int; tmin::Float64=0.2,tmax::Float64=1.0,clicktot::Int=40)
    
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
        pz::Vector{TT};dt::Float64=1e-4) where {TT <: Any}
    
    vari, B, lambda, vara, vars, phi, tau_phi = pz
    
    La, Ra = make_adapted_clicks(pz,L,R)
    
    nT = Int(ceil(T/dt)) # number of timesteps
    t = 0.:dt:nT*dt-dt
    hereL, hereR = vec(qfind(t,L)), vec(qfind(t,R))

    A = Vector{TT}(undef,nT)
    a = sqrt(vari)*randn()

    for t = 1:nT

        #inputs
        any(t .== hereL) ? sL = sum(La[t .== hereL]) : sL = zero(TT)
        any(t .== hereR) ? sR = sum(Ra[t .== hereR]) : sR = zero(TT)
        var, mu = vars * (sL + sR), -sL + sR  
        
        (sL + sR) > zero(TT) ? a += mu + sqrt(var) * randn() : nothing
        a += (dt*lambda) * a + sqrt(vara * dt) * randn()

        abs(a) > B ? (a = B * sign(a); A[t:nT] .= a; break) : A[t] = a

    end               
    
    return A
    
end