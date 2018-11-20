module global_functions

using StatsBase

export diffLR, latent_and_spike_params
export map_latent_params!, inv_map_latent_params!, map_sig_params!, inv_map_sig_params!
export make_adapted_clicks
export my_sigmoid

function make_adapted_clicks{TT}(phi::TT, tau_phi::TT, 
        L::Vector{Float64}, R::Vector{Float64})

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

function adapt_clicks!{TT}(Ca::Vector{TT}, phi::TT, tau_phi::TT, ici::Vector{Float64})
    
    for i = 1:length(ici)
        arg = abs(1. - Ca[i]*phi)
        arg > 1e-150 ? Ca[i+1] = 1. - exp((-ici[i] + tau_phi*log(arg))/tau_phi) : nothing
    end
    
end

function my_sigmoid(x,p)
    
    temp = -p[3]*x + p[4]

    y = p[1] + p[2]./(1. + exp.(temp));
     
    #protect from NaN gradient values
    y[exp.(temp) .<= 1e-150] = p[1] + p[2]
    y[exp.(temp) .>= 1e150] = p[1]
    
    return y
    
end

function map_sig_params!{TT}(py::Vector{TT},map_str::String)
        
    if map_str == "exp"
        py[1:2] = exp.(py[1:2])
        py[3:4] = py[3:4]
    elseif map_str == "tanh"
        py[1:2] = 1e-5 + 99.99 * 0.5*(1+tanh.(py[1:2]))
        py[3:4] = -9.99 + 9.99*2 * 0.5*(1+tanh.(py[3:4]))
    end
    
    return py
    
end

function inv_map_sig_params!{TT}(py::Vector{TT},map_str::String)
    
    if map_str == "exp"
        py[1:2] = log.(py[1:2])
        py[3:4] = py[3:4]
    elseif map_str == "tanh"
        py[1:2] = atanh.(((py[1:2] - 1e-5)/(99.99*0.5)) - 1)
        py[3:4] = atanh.(((py[3:4] + 9.99)/(9.99*2*0.5)) - 1)
    end
    
    return py
    
end

function map_latent_params!(x,map_str,dt)
    
    x[[1,5,6]] = exp.(x[[1,5,6]]);
    x[2] = 0.5*(1+tanh(x[2]));
    
    if map_str == "exp"
        x[3] = 2. + exp(x[3]);
    elseif map_str == "tanh"
        x[3] = 2 + 100 * 0.5*(1+tanh.(x[3]))
    end
    
    x[4] = -1./(2*dt) + (1./dt)*(0.5*(1.+tanh(x[4])));
    x[7] = exp(x[7]);
    x[8] = exp(x[8]);
    
    return x
    
end

function inv_map_latent_params!(x,map_str,dt)
    
    x[[1,5,6]] = log.(x[[1,5,6]]);
    x[2] = atanh(2.*x[2]-1.);
    
    if map_str == "exp"
        x[3] = log(x[3]-2.);
    elseif map_str == "tanh"
        x[3] = atanh.(((x[3] - 2.)/(100*0.5))-1)
    end
    
    x[4] = atanh((2 .* dt * (x[4] + 1./(2.*dt))) - 1.);
    x[7] = log(x[7]);
    x[8] = log(x[8]);
    
    return x
    
end

function latent_and_spike_params{TT}(p_opt::Vector{TT}, p_const::Vector{Float64}, fit_vec::BitArray{1}, model_type::String)
    
    p = Vector{TT}(length(fit_vec))
    p[fit_vec] = p_opt;
    p[.!fit_vec] = p_const;
    
    if any(model_type .== "spikes") & any(model_type .== "choice")
 
        pz = p[1:8];
        bias = p[9];
        
        pytemp = reshape(p[10:end],4,:)
        py = Vector{Vector{TT}}(0)
        
        for i = 1:size(pytemp,2)
            push!(py,pytemp[:,i])
        end
               
        return pz,py,bias
    
    elseif any(model_type .== "spikes") 
                
        pz = p[1:8];

        pytemp = reshape(p[9:end],4,:)
        py = Vector{Vector{TT}}(0)

        for i = 1:size(pytemp,2)
            push!(py,pytemp[:,i])
        end

        return pz, py

    elseif any(model_type .== "choice")
                
        pz = p[1:8];
        bias = p[9];
                
        return pz, bias
        
    end
    
end

function break_params{TT}(p::Vector{TT}, fit_vec::BitArray{1})
    
    p_opt = p[fit_vec];
    p_const = Array{Float64}(p[.!fit_vec]);
    
    return p_opt, p_const
    
end

#should modify this to return the differnece only, and then allow filtering afterwards

function diffLR(nT,L,R;path=false,dt::Float64=2e-2)
    
    #compute the cumulative diff of clicks
    t = 0:dt:nT*dt;
    L = fit(Histogram,L,t,closed=:left)
    R = fit(Histogram,R,t,closed=:left)
    
    if path
        diffLR = cumsum(-L.weights + R.weights)
    else
        diffLR = sum(-L.weights + R.weights)
    end
    
end

end
