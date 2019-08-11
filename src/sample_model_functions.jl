
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
    
    T = Int(ceil(T/2e-2)) * 2e-2
    
    return T,R,L
    
end

function sample_latent(nT::Int, L::Vector{Float64},R::Vector{Float64},
        nL::Vector{Int}, nR::Vector{Int}, 
        pz::Vector{TT}, use_bin_center::Bool; 
        dt::Float64=1e-4) where {TT <: Any}
    
    vari, B, lambda, vara, vars, phi, tau_phi = pz
    
    La, Ra = make_adapted_clicks(pz,L,R)

    A = Vector{TT}(undef,nT)
    a = sqrt(vari)*randn()

    for t = 1:nT

        #inputs
        #any(t .== nL) ? sL = sum(La[t .== nL]) : sL = zero(TT)
        #any(t .== nR) ? sR = sum(Ra[t .== nR]) : sR = zero(TT)
        #var, mu = vars * (sL + sR), -sL + sR  
        
        #(sL + sR) > zero(TT) ? a += mu + sqrt(var) * randn() : nothing
        #a += (dt*lambda) * a + sqrt(vara * dt) * randn()
        
        #if (vars <= eps(TT)) && (vara <= eps(TT))
            
            #dt == 1e-4 ? (@warn "yo") : nothing
            #this is to deal with determinstic models, when data is not being generated and dt is larger, to handle the first bin
            if use_bin_center && t == 1
            
                a = sample_one_step!(a, t, vara, vars, lambda, nL, nR, La, Ra, dt/2)
                #@warn "yo"
                #eta = sqrt(vara * dt/2 + var) * randn()
                #if abs(lambda) < 1e-150 
                #    a += mu + eta
                #else
                #    h = mu/(dt/2*lambda)
                #    a = exp(lambda*dt/2)*(a + h) - h + eta
                #end
            else
            
                a = sample_one_step!(a, t, vara, vars, lambda, nL, nR, La, Ra, dt)


                #eta = sqrt(vara * dt + var) * randn()
                #if abs(lambda) < 1e-150 
                #    a += mu + eta
                #else
                #    h = mu/(dt*lambda)
                #    a = exp(lambda*dt)*(a + h) - h + eta
                #end
            end
        #else
            #this input should decay too, no, like above?
            #can i just add noise to determinisitc solution
            #(sL + sR) > zero(TT) ? I = mu + sqrt(var) * randn() : I = 0.
            #a += (dt*lambda) * a + sqrt(vara * dt) * randn() + I
        #end

        abs(a) > B ? (a = B * sign(a); A[t:nT] .= a; break) : A[t] = a

    end               
    
    return A
    
end

function sample_one_step!(a::TT, t::Int, vara::TT, vars::TT, lambda::TT, nL::Vector{Int}, nR::Vector{Int}, 
        La, Ra, dt::Float64) where {TT <: Any}
    
    #inputs
    any(t .== nL) ? sL = sum(La[t .== nL]) : sL = zero(TT)
    any(t .== nR) ? sR = sum(Ra[t .== nR]) : sR = zero(TT)
    var, mu = vars * (sL + sR), -sL + sR  
    
    eta = sqrt(vara * dt + var) * randn()
    
    if abs(lambda) < 1e-150 
        a += mu + eta
    else
        h = mu/(dt*lambda)
        a = exp(lambda*dt)*(a + h) - h + eta
    end
    
    return a

end