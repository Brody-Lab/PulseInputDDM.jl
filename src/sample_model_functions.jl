
function sample_clicks(ntrials::Int; rng::Int=1)
    
    Random.seed!(rng)
    
    data = Dict()

    output = map(generate_stimulus,1:ntrials)
    #output = map(i-> generate_stimulus(i;tmin=0.5, tmax=0.7),1:ntrials)
    #@warn "all trials 1.0 seconds"
    #@warn "all trials divisible by 0.01, moved"
    #@warn "all normal"
    #@warn "0.6 or 0.8"
    @warn "rounded, all times"
    #@warn "not divied, all times"

    data["leftbups"] = map(i->output[i][3],1:ntrials)
    data["rightbups"] = map(i->output[i][2],1:ntrials)
    data["T"] = map(i->output[i][1],1:ntrials)
    data["ntrials"] = ntrials
    
    return data
    
end

function generate_stimulus(i::Int; tmin::Float64=0.2,tmax::Float64=1.0,clicktot::Int=40)
    
    T = tmin + (tmax-tmin)*rand()
    #T = [0.6712345982,0.819045238761][1+Int(round(rand()))]
    #T = [0.67, 0.82][1+Int(round(rand()))]
    #T = [0.6712, 0.8190][1+Int(round(rand()))]
    #T = [0.6, 0.8][1+Int(round(rand()))]
    
    ratetot = clicktot/T
    Rbar = ratetot*rand()
    Lbar = ratetot - Rbar

    R = cumsum(rand(Exponential(1/Rbar),clicktot))
    L = cumsum(rand(Exponential(1/Lbar),clicktot))
    R = vcat(0,R[R .<= T])
    L = vcat(0,L[L .<= T])
    
    T = Int(ceil(T/1e-2)) * 1e-2
    
    return T,R,L
    
end

function sample_latent(nT::Int, L::Vector{Float64},R::Vector{Float64},
        nL::Vector{Int}, nR::Vector{Int}, 
        pz::Vector{TT}; dt::Float64=1e-4) where {TT <: Any}
    
    vari, B, lambda, vara, vars, phi, tau_phi = pz
    
    La, Ra = make_adapted_clicks(pz,L,R)

    A = Vector{TT}(undef,nT)
    a = sqrt(vari)*randn()

    for t = 1:nT

        #inputs
        any(t .== nL) ? sL = sum(La[t .== nL]) : sL = zero(TT)
        any(t .== nR) ? sR = sum(Ra[t .== nR]) : sR = zero(TT)
        var, mu = vars * (sL + sR), -sL + sR  
        
        #(sL + sR) > zero(TT) ? a += mu + sqrt(var) * randn() : nothing
        #a += (dt*lambda) * a + sqrt(vara * dt) * randn()
        
        if (vars == zero(TT)) && (vara == zero(TT))                
            h = mu/(dt*lambda)
            a = exp(lambda*dt)*(a + h) - h
        else
            (sL + sR) > zero(TT) ? I = mu + sqrt(var) * randn() : I = 0.
            a += (dt*lambda) * a + sqrt(vara * dt) * randn() + I
        end

        abs(a) > B ? (a = B * sign(a); A[t:nT] .= a; break) : A[t] = a

    end               
    
    return A
    
end