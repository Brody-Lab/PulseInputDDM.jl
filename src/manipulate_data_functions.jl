#################################### Choice observation model #################################

#check about this and other make_data functions to make sure they're OK

function make_data(path::String, sessids::Vector{Vector{Int}}, ratnames::Vector{String}, dt::Float64)

    data = Dict("leftbups" => Vector{Vector{Float64}}(), "rightbups" => Vector{Vector{Float64}}(), 
                "binned_leftbups" => Vector{Vector{Int64}}(), "binned_rightbups" => Vector{Vector{Int64}}(),
                "T" => Vector{Float64}(), "nT" => Vector{Int64}(), 
                "pokedR" => Vector{Bool}(), "correct_dir" => Vector{Bool}(), 
                "sessid" => Vector{Int}(), "ratname" => Vector{String}(),
                "dt" => dt);
    
    for j = 1:length(ratnames)
        for i = 1:length(sessids[j])
            file = matopen(path*"/"*ratnames[j]*"_"*string(sessids[j][i])*".mat")
            rawdata = read(file,"rawdata")
            data = package_data!(data,rawdata,ratnames[j],dt)
        end
    end
    
    return data
    
end

function package_data!(data::Dict,rawdata::Dict,ratname::String,dt::Float64=1e-2)

    ntrials = length(rawdata["T"])
    binnedT = ceil.(Int,rawdata["T"]/dt);

    append!(data["T"],rawdata["T"])
    append!(data["nT"],binnedT)
    append!(data["pokedR"],vec(convert(BitArray,rawdata["pokedR"])))
    append!(data["correct_dir"],vec(convert(BitArray,rawdata["correct_dir"])))
    
    append!(data["leftbups"],map(x->vec(collect(x)),rawdata["leftbups"]))
    append!(data["rightbups"],map(x->vec(collect(x)),rawdata["rightbups"]))
    append!(data["binned_leftbups"],map((x,y)->vec(qfind(0.:dt:x*dt,y)),binnedT,rawdata["leftbups"]))
    append!(data["binned_rightbups"],map((x,y)->vec(qfind(0.:dt:x*dt,y)),binnedT,rawdata["rightbups"]))
    append!(data["sessid"],map(x->x[1],rawdata["sessid"]))
    append!(data["ratname"],map(x->ratname,rawdata["sessid"]))

    return data

end

#################################### Poisson neural observation model #########################

function make_data(path::String,sessids::Vector{Vector{Int}},ratnames::Vector{String};
        model_type::String="spikes",dt::Float64=1e-2,organize::String="by_trial",shifted::Bool=false)

    data = Dict("leftbups" => Vector{Vector{Float64}}(), "rightbups" => Vector{Vector{Float64}}(), 
                "binned_leftbups" => Vector{Vector{Int64}}(), "binned_rightbups" => Vector{Vector{Int64}}(),
                "T" => Vector{Float64}(), "nT" => Vector{Int64}(), 
                "pokedR" => Vector{Bool}(), "correct_dir" => Vector{Bool}(), 
                "spike_counts" => Vector{Vector{Vector{Int64}}}(),"N" => Vector{Vector{Int64}}(),
                "trial" => Vector{UnitRange{Int64}}(),
                "sessid" => Vector{Int}(), "N0" => 0, "trial0" => 0,
                "cell" => Vector{Vector{Int}}(),"ratname" => Vector{String}());
    
    for j = 1:length(ratnames)
        for i = 1:length(sessids[j])
            if shifted
                file = matopen(path*"/shifted_"*ratnames[j]*"_"*string(sessids[j][i])*".mat")
            else
                file = matopen(path*"/"*ratnames[j]*"_"*string(sessids[j][i])*".mat")
            end
            rawdata = read(file,"rawdata")
            data = package_data!(data,rawdata,model_type,ratnames[j],dt=dt,organize=organize)
        end
    end

    N = data["N0"]
    data["dt"] = dt
    
    return data, N
    
end

function package_data!(data,rawdata,model_type::String,ratname::String;dt::Float64=2e-2,organize::String="by_trial")

    ntrials = length(rawdata["T"])
    binnedT = ceil.(Int,rawdata["T"]/dt);

    append!(data["T"],rawdata["T"])
    append!(data["nT"],binnedT)
    append!(data["pokedR"],vec(convert(BitArray,rawdata["pokedR"])))
    append!(data["correct_dir"],vec(convert(BitArray,rawdata["correct_dir"])))
    
    append!(data["leftbups"],map(x->vec(collect(x)),rawdata["leftbups"]))
    append!(data["rightbups"],map(x->vec(collect(x)),rawdata["rightbups"]))
    append!(data["binned_leftbups"],map((x,y)->vec(qfind(0.:dt:x*dt,y)),binnedT,rawdata["leftbups"]))
    append!(data["binned_rightbups"],map((x,y)->vec(qfind(0.:dt:x*dt,y)),binnedT,rawdata["rightbups"]))
    append!(data["sessid"],map(x->x[1],rawdata["sessid"]))
    append!(data["cell"],map(x->vec(collect(x)),rawdata["cell"]))
    append!(data["ratname"],map(x->ratname,rawdata["cell"]))
    
    if any(model_type .== "spikes")
        
        N = size(rawdata["St"][1],2)

        if organize == "by_trial"
    
            append!(data["spike_counts"],map((x,y)->map(z->fit(Histogram,vec(collect(y[z])), 
                    0.:dt:x*dt,closed=:left).weights,1:N),binnedT,rawdata["St"]))
            append!(data["N"],repeat([collect(data["N0"]+1:data["N0"]+N)],inner=ntrials))
            #added 2/5 to note which trials have which neurons
            append!(data["trial"],repeat([data["trial0"]+1:data["trial0"]+ntrials],inner=N))

        elseif organize == "by_neuron"
            
            append!(data["spike_counts"],map!(z -> map!((x,y) -> fit(Histogram,vec(collect(y[z])), 
                    0.:dt:x*dt,closed=:left).weights,Vector{Vector}(undef,ntrials),
                    binnedT,rawdata["St"]),Vector{Vector}(undef,N),1:N));
            append!(data["trial"],repeat([data["trial0"]+1:data["trial0"]+ntrials],inner=N));

        end
        
        data["N0"] += N
        data["trial0"] += ntrials

    end

    return data

end

#################################### OLD #########################

function load_data(path::String,model_type::Union{String,Array{String}},
        reload_pth::String,map_str::String,ratname::String)
    
    initials = reload_pth*"/initials.jld"
    
    if isfile(initials)
        
        fit_vec, data, model_type, p0_z =  load(initials, "fit_vec", "data", "model_type", "p0_z")
        
        if any(model_type .== "spikes") 
            
            p0_y, beta_y, mu0_y =  load(initials,"p0_y", "beta_y", "mu0_y");
            
            return fit_vec, data, model_type, p0_z, p0_y, beta_y, mu0_y 
            
        elseif any(model_type .== "choice")
            
            p0_bias = load(initials,"p0_bias");
            
            return fit_vec, data, model_type, p0_z, p0_bias 

        end

    else
                
        sessid = get_sessid(ratname)
        
        # should consider changing this at some point because it's often not clear if a value
        # is a constant, or just the initial value
        
        #          vari       inatt        B    lambda       vara    vars     phi    tau_phi 
        fit_vec = [falses(1);falses(1);    trues(4);                          falses(2)];
        p0_z =    [1e-6,      0.,         20., 1e-3,        10.,    1.,      1.,    0.2]
    
        if any(model_type .== "choice") 
            p0_bias = 1e-6
            fit_vec = cat(1,fit_vec,trues(1))
        end
        
        if any(model_type .== "spikes")

            data,N = make_data(path,sessid,ratname;dt=1e-3,organize="by_neuron");  
            
            beta_y = vcat(1e-6*ones(4))
            mu0_y = map(x->vcat(x,zeros(3)),zeros(N))
            
            p0_y = x0_spikes(data,map_str,beta_y,mu0_y,dt=1e-3)
            fit_vec = cat(1,fit_vec,trues(dimy*N))
            
        end  
        
        data,N = make_data(path,sessid,ratname;dt=dt,organize="by_trial");
        
        if any(model_type .== "spikes") & any(model_type .== "choice")
            
            save(initials,
                "fit_vec",fit_vec,"data",data,"sessid",sessid,"model_type",model_type,
                "p0_z",p0_z,"p0_bias",p0_bias,"p0_y",p0_y,"beta_y",beta_y,"mu0_y",mu0_y);
            
            return fit_vec, data, model_type, p0_z, p0_bias, p0_y, beta_y, mu0_y

        elseif any(model_type .== "spikes") 
            
            save(initials,
                "fit_vec",fit_vec,"data",data,"sessid",sessid,"model_type",model_type,
                "p0_z",p0_z,"p0_y",p0_y,"beta_y",beta_y,"mu0_y",mu0_y);
            
            return fit_vec, data, model_type, p0_z, p0_y, beta_y, mu0_y
            
        elseif any(model_type .== "choice")
            
            save(initials,
                "fit_vec",fit_vec,"data",data,"sessid",sessid,"model_type",model_type,
                "p0_z",p0_z,"p0_bias",p0_bias);
            
            return fit_vec, data, model_type, p0_z, p0_bias

        end

    end
        
end

function get_sessid(ratname::String)
    
    if ratname == "B053"
        sessid = [51344, 52162,52425, 53154,54377]    
    elseif ratname == "B068"
        sessid = [46331,46484,46630,47745,48117,48988,49313,49380,49545,49819,50353,50838,50951,51621,51852,52307]
    elseif ratname == "T011"
        #sessid = [153219,153382,153510,154806,154950,155375,155816,155954,157026,157178,157483,158231,161057,161351,164574,165972]   
        sessid = [153219,153382,153510,154806,154950,155375,155816,155954,157026,157178,157483]   
    elseif ratname == "T034"
        #sessid = [151801,152157,152272,152370,152509,152967,153253,153859,154807,154949,
        #    157482,161543,163862,163993,164420,164573,165023,168608,169683];
        sessid = [153253,154807,154949,157482,164573,169683];
    elseif ratname == "T035"
        #sessid = [152177,153274,153536,154288,154440,155839,156150,156443,156896,
        #    157200,157359,161394,162142, 162258,163098,163885,164449,164604,164752,164900,165058,166135,
        #    166590,167725,167855,167993,168132,168628,169448,169736,169873,169993]
        sessid = [169448,167725,166135,164900]
    elseif ratname == "T036"
        #sessid = [154154,154291,154448,154991,155124,155247,155840,157201,157357,157507,168499,168627]
        #sessid = [157201,157357,157507,168499]
        sessid = [157201]
    elseif ratname == "T063"
        sessid = [191956,193936,194554,194649,194770,194898,195271,195546,195676,195791,196336,196580,
            196708,197075,197212,197479,198004]
    elseif ratname == "T068"
        sessid = [195545,195790,196335,196579,196709,197204,197478,198003,198137,198249]
    end
    
end

function package_extended_data!(data,rawdata,model_type::String,ratname,ts::Float64;dt::Float64=2e-2,organize::String="by_trial")

    ntrials = length(rawdata["T"])
    
    append!(data["sessid"],map(x->x[1],rawdata["sessid"]))
    append!(data["cell"],map(x->vec(collect(x)),rawdata["cell"]))
    append!(data["ratname"],map(x->ratname,rawdata["cell"]))
    
    maxT = ceil.(Int,(rawdata["T"])/dt)
    binnedT = ceil.(Int,(rawdata["T"] + ts)/dt);

    append!(data["nT"],binnedT)
    
    if any(model_type .== "spikes")
        
        N = size(rawdata["St"][1],2)

        if organize == "by_trial"
    
            append!(data["spike_counts"],map((x,y)->map(z->fit(Histogram,vec(collect(y[z])), 
                    -ts:dt:x*dt,closed=:left).weights,1:N),maxT,rawdata["St"]));
            append!(data["N"],repmat([collect(data["N0"]+1:data["N0"]+N)],ntrials));

        elseif organize == "by_neuron"
            
            append!(data["spike_counts"],map!(z -> map!((x,y) -> fit(Histogram,vec(collect(y[z])), 
                    -ts:dt:x*dt,closed=:left).weights,Vector{Vector}(ntrials),
                    maxT,rawdata["St"]),Vector{Vector}(N),1:N));
            append!(data["trial"],repmat([data["trial0"]+1:data["trial0"]+ntrials],N));

        end
        
        data["N0"] += N
        data["trial0"] += ntrials

    end

    return data

end

#scrub a larger dataset to only keep data relevant to a single neuron
function keep_single_neuron_data!(data,i)
    
    data["nT"] = data["nT"][data["trial"][i]]
    data["leftbups"] = data["leftbups"][data["trial"][i]]
    data["rightbups"] = data["rightbups"][data["trial"][i]]
    data["binned_rightbups"] = data["binned_rightbups"][data["trial"][i]]
    data["binned_leftbups"] = data["binned_leftbups"][data["trial"][i]]

    data["N"] = data["N"][data["trial"][i]]
    data["spike_counts"] = data["spike_counts"][data["trial"][i]]

    data["spike_counts"] = map((x,y)->x = x[y.==i],data["spike_counts"],data["N"])
    map!(x->x = [1],data["N"],data["N"])
    
    return data
    
end

#just hanging on to this for some later time
function my_callback(os)

    so_far = time() - start_time
    println(" * Time so far:     ", so_far)
  
    history = Array{Float64,2}(sum(fit_vec),0)
    history_gx = Array{Float64,2}(sum(fit_vec),0)
    for i = 1:length(os)
        ptemp = group_params(os[i].metadata["x"], p_const, fit_vec)
        ptemp = map_func!(ptemp,model_type,"tanh",N=N)
        ptemp_opt, = break_params(ptemp, fit_vec)       
        history = cat(2,history,ptemp_opt)
        history_gx = cat(2,history_gx,os[i].metadata["g(x)"])
    end
    save(out_pth*"/history.jld", "history", history, "history_gx", history_gx)

    return false

end

function group_by_neuron(data)
    
    trials = Vector{Vector{Int}}()
    SC = Vector{Vector{Vector{Int64}}}()

    map(x->push!(trials,Vector{Int}(undef,0)),1:data["N0"])
    map(x->push!(SC,Vector{Vector{Int}}(undef,0)),1:data["N0"])

    map(y->map(x->push!(trials[x],y),data["N"][y]),1:data["trial0"])
    map(n->map(t->append!(SC[n],data["spike_counts"][t][data["N"][t] .== n]),
        trials[n]),1:data["N0"])
    
    return trials, SC
    
end

#function my_callback(os)

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
    #print(os[1]["x"])
    #save(ENV["HOME"]*"/spike-data_latent-accum"*"/history.jld", "os", os)
    #print(path)

#    return false

#end

##############################################################################################################

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