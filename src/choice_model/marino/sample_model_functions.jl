
#################################### Choice observation model #################################

function sample_inputs_and_choices(pz::Vector{Float64}, pd::Vector{Float64}, pw::Vector{Float64}, ntrials::Int; 
        dtMC::Float64=1e-4, rng::Int = 1, use_bin_center::Bool=false)
    
    data = sample_clicks(ntrials)         
    data = sample_choices_all_trials!(data, pz, pd, pw; dtMC=dtMC, rng=rng, use_bin_center=use_bin_center)
            
    return data
    
end

function sample_choices_all_trials!(data::Dict, pz::Vector{Float64}, pd::Vector{Float64}, pw::Vector{Float64}; 
        dtMC::Float64=1e-4, rng::Int = 1, use_bin_center::Bool=false)
            
    Random.seed!(rng)
    nT,nL_l,nR_l = bin_clicks(data["T"],data["leftbups_loc"],data["rightbups_loc"],dtMC,use_bin_center)
    nT,nL_f,nR_f = bin_clicks(data["T"],data["leftbups_freq"],data["rightbups_freq"],dtMC,use_bin_center)
        
    data["pokedR"] = pmap((nT,L_l,R_l,nL_l,nR_l,L_f,R_f,nL_f,nR_f,context,rng) -> 
        sample_choice_single_trial(nT,L_l,R_l,nL_l,nR_l,
            L_f,R_f,nL_f,nR_f,
            pz,pd,pw,context;
            rng=rng), nT, data["leftbups_loc"], data["rightbups_loc"], nL_l, nR_l, 
            data["leftbups_freq"], data["rightbups_freq"], nL_f, nR_f, data["context_loc"],
            shuffle(1:length(data["T"])))
            
    return data
    
end

function sample_choice_single_trial(nT::Int, L_l::Vector{Float64}, R_l::Vector{Float64},
        nL_l::Vector{Int}, nR_l::Vector{Int},
        L_f::Vector{Float64}, R_f::Vector{Float64},
        nL_f::Vector{Int}, nR_f::Vector{Int},
        pz::Vector{Float64},pd::Vector{Float64},pw::Vector{Float64},context::Bool;
        dtMC::Float64=1e-4,rng::Int=1)
    
    Random.seed!(rng)
    a = sample_latent(nT,L_l,R_l,nL_l,nR_l,
        L_f,R_f,nL_f,nR_f,pz,pw,context;dt=dtMC)
    
    bias,lapse = pd[1],pd[2]
            
    rand() > lapse ? choice = a[end] >= bias : choice = Bool(round(rand()))
    
end

function sample_latent(nT::Int, L_l::Vector{Float64},R_l::Vector{Float64},
        nL_l::Vector{Int}, nR_l::Vector{Int},
        L_f::Vector{Float64},R_f::Vector{Float64},
        nL_f::Vector{Int}, nR_f::Vector{Int},
        pz::Vector{TT}, pw::Vector{TT},context::Bool; 
        dt::Float64=1e-4) where {TT <: Any}
    
    vari, B, lambda, vara, vars, phi, tau_phi = pz
    
    La_l, Ra_l = make_adapted_clicks(pz,L_l,R_l)
    La_f, Ra_f = make_adapted_clicks(pz,L_f,R_f)
    
    if context #location context is true...
        w_loc, w_freq = pw[1:2] #...set location weight to active
    else #if location context if false, i.e. freq context
        w_freq, w_loc = pw[1:2] #...set freq weight to active
    end

    A = Vector{TT}(undef,nT)
    a = sqrt(vari)*randn()

    for t = 1:nT
            
        a = sample_one_step!(a, t, vara, vars, lambda, nL_l, nR_l, nL_f, nR_f, La_l, Ra_l, La_f, Ra_f, dt, w_loc, w_freq)

        abs(a) > B ? (a = B * sign(a); A[t:nT] .= a; break) : A[t] = a

    end               
    
    return A
    
end

function sample_one_step!(a::TT, t::Int, vara::TT, vars::TT, lambda::TT, nL_l::Vector{Int}, nR_l::Vector{Int}, 
        nL_f::Vector{Int}, nR_f::Vector{Int}, La_l, Ra_l, La_f, Ra_f, dt::Float64, w_loc, w_freq) where {TT <: Any}
    
    #inputs
    #any(t .== nL) ? sL = sum(La[t .== nL]) : sL = zero(TT)
    #any(t .== nR) ? sR = sum(Ra[t .== nR]) : sR = zero(TT)
    #var, mu = vars * (sL + sR), -sL + sR  
    
    sum_clicks, mu = make_inputs(t, nL_l, nR_l, nL_f, nR_f, La_l, Ra_l, La_f, Ra_f, w_loc, w_freq)
    
    var = vars * sum_clicks
    
    eta = sqrt(vara * dt + var) * randn()
    
    if abs(lambda) < 1e-150 
        a += mu + eta
    else
        h = mu/(dt*lambda)
        a = exp(lambda*dt)*(a + h) - h + eta
    end
    
    return a

end

function make_inputs(t, nL_l, nR_l, nL_f, nR_f, La_l, Ra_l, La_f, Ra_f, w_loc::TT, w_freq::TT) where {TT <: Any}
    
    any(t .== nL_l) ? sL_l = sum(La_l[t .== nL_l]) : sL_l = zero(TT)
    any(t .== nR_l) ? sR_l = sum(Ra_l[t .== nR_l]) : sR_l = zero(TT)
    
    any(t .== nL_f) ? sL_f = sum(La_f[t .== nL_f]) : sL_f = zero(TT)
    any(t .== nR_f) ? sR_f = sum(Ra_f[t .== nR_f]) : sR_f = zero(TT)

    sum_clicks = (w_loc * (sL_l + sR_l) + w_freq * (sL_f + sR_f))
    mu = w_loc * (-sL_l + sR_l) + w_freq * (-sL_f + sR_f)
    
    return sum_clicks, mu
    
end