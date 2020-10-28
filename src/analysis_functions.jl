# requires Colors, Parameters, PyPlot, MAT, Bootstrap, 
# and Statistics: mean, median, std, quantile

function return_fit_params(ratname, identifier; prmfile_num = 1)
     # loading parameter file
    prmpath = "pkg_data/example_fits/"
    str = Regex("sess_rawdata_"*ratname*".*"*identifier*".mat")
    prmfiles = filter(x->occursin(str,x),readdir(prmpath))
    print(prmfiles[prmfile_num])
    x, modeltype = read(matopen(prmpath*prmfiles[prmfile_num]),"ML_params","modeltype")
    θ = reconstruct_model(x, modeltype)
    
    return θ

end


function load_data_and_param(ratname, identifier; use_data_initpt = false, prmfile_num = 1)
    
    # loading parameter file
    prmpath = "pkg_data/example_fits/"
    str = Regex("sess_rawdata_"*ratname*".*"*identifier*".mat")
    prmfiles = filter(x->occursin(str,x),readdir(prmpath))
    print(prmfiles[prmfile_num])
    
    x, modeltype = read(matopen(prmpath*prmfiles[prmfile_num]),"ML_params","modeltype")
    
    if identifier == "expfilter_ce_lr_ndmod_wleak_hlapse" # poor man backward compatibility
        x = vcat(x[1:12], 0., x[13:end])
        x[16] = x[16] + x[19]
        x[17] = x[17] + x[19]
        x[20] = x[20] - x[19]
        x = vcat(x[1:18], x[20:end])
    elseif identifier == "expfilter_ce_lr_red_ndmod_lapsemod_oct6"  # lapse adjustment (exponential)
        x[11] = x[11] - 0.05
         x[12] = 0.
    end
    θ = reconstruct_model(x, modeltype)
    
    # loading data with 5s long stimulus for simulating
    datapath = "pkg_data/"
    datafile = "sess_rawdata_"*ratname*"_wholeStim.mat"
    inp, simdict = load(datapath*datafile, sim = true, dt = 1e-3)
    
    if use_data_initpt == false
        choices, RT = rand(θ, inp, simdict, rng = abs(rand(Int)))
    else
        inp_temp, simdict_temp = load(datapath*datafile, sim = false, dt = 1e-3)
        a_0 = compute_initial_pt(θ.hist_θz, θ.base_θz.B0, simdict_temp)
        choices, RT = rand(θ, inp, simdict, rng = abs(rand(Int)), ipt = a_0)
    end
     
    # repacking
    dt = simdict["dt"] 
    centered = inp[1].centered
    clicks = map(x-> x.clicks,inp)
    map((clicks, RT) -> clicks.T = round(RT, digits =length(string(dt))-2), clicks, RT)
    binned_clicks = bin_clicks.(clicks, centered=centered, dt=dt)
    inputs = choiceinputs.(clicks, binned_clicks, dt, centered)
    simdata = choicedata.(inputs, choices, simdict["sessbnd"])
    
    if use_data_initpt
        return θ, simdata, simdict, a_0
    else
        return θ, simdata, simdict
    end
end


function get_gamma(data)
    gamma = map(x-> x.click_data.clicks.gamma, data)
    ugam = sort(unique(gamma)) 
    return gamma, ugam
end


function unpack_data(data, data_dict; gethist = false)
    pokedR = map(x-> x.choice, data)
    RT = map(x-> x.click_data.clicks.T, data)
    hits = convert(Array{Bool}, pokedR .== data_dict["correct"])   
    if gethist
        posthit = vcat(0, hits[1:end-1])
        postR   = vcat(0, pokedR[1:end-1])
        return pokedR, RT, hits, posthit, postR
    else
        return pokedR, RT, hits
    end
end


function get_psychometric(pR, gamma)
    ugam = sort(unique(gamma))   
    fracR =  compute_cond_bootci(pR, gamma, "mean"; nboot = 1000, conf = 0.95)
    return fracR
end


function get_meanRTs(RT, hits, gamma; stat = "mean")
    ugam = sort(unique(gamma)) 
    meanRT = compute_cond_bootci(RT, gamma, stat)
    hitRT = compute_cond_bootci(RT[hits .== 1], gamma[hits .== 1], stat)
    errRT = compute_cond_bootci(RT[hits .== 0], gamma[hits .== 0], stat)
    return meanRT, hitRT, errRT
end


function get_postsummary(data, data_dict, X; stat = "mean")
    gamma, ugam = get_gamma(data)
    pR, RT, hit = unpack_data(data, data_dict)
    if X == "RT"
        summary = get_postsummary(pR, RT, hit, RT, gamma, stat = stat)
    else
        summary = get_postsummary(pR, RT, hit, pR, gamma, stat = "mean")
    end
    return summary
end


function get_postsummary(pokedR, RT, hits, X, gamma; stat = "mean")
    ugam = sort(unique(gamma))   
    pH = vcat(0, hits[1:end-1])
    pR  = vcat(0, pokedR[1:end-1])  # POST RIGHT HERE
    
    postRcorr = compute_cond_bootci(X[pH .== 1 .& pR .== 1], gamma[pH .== 1 .& pR .== 1], stat)
    postLcorr = compute_cond_bootci(X[pH .== 1 .& .~pR .== 1], gamma[pH .== 1 .& .~pR .== 1], stat)
    postcorr = compute_cond_bootci(X[pH .== 1], gamma[pH .== 1], stat)
    
    postRerr = compute_cond_bootci(X[pH .!= 1 .& pR .== 1], gamma[pH .!= 1 .& pR .== 1], stat)
    postLerr = compute_cond_bootci(X[pH .!= 1 .& .~pR .== 1], gamma[pH .!= 1 .& .~pR .== 1], stat)
    posterr = compute_cond_bootci(X[pH .== 0], gamma[pH .== 0], stat)
    
    return Dict("postRcorr" => postRcorr, "postLcorr" => postLcorr, 
                "postcorr" => postcorr, "posterr" => posterr,
                "postRerr" => postRerr, "postLerr" => postLerr) 
end


function get_errorbars(X)
   return transpose(abs.(X[:,2:3] .- X[:,1]))    
end


function compute_cond_bootci(var, cond, stat; nboot = 1000, conf = 0.95)
    
    # defining which statistic to compute ci over
    s = Symbol(stat)
    f = getfield(Main,s)
    
    ucond = sort(unique(cond))
    ci = zeros(length(ucond),3)
    
    for i = 1:length(ucond)
        bc = bootstrap(f, var[cond .== ucond[i]], BasicSampling(nboot))
        tmp = confint(bc, BasicConfInt(conf))
        ci[i,:] = map(i->tmp[1][i],1:3)
    end
    return ci
end

"""
Plotting-related functions

"""
function set_plot(axs, xtickslab; xlab = "Stimulus strength", ylab= "Fraction chose right")
    fontn = "Helvetica"
    tickfontsize = 15
    axs.spines["top"].set_visible(false)
    axs.spines["right"].set_visible(false)
    axs.set_xticks(round.(xtickslab, digits =1)) 
    axs.set_xlabel(xlab, fontsize = tickfontsize, fontname=fontn)
    axs.set_ylabel(ylab, fontsize = tickfontsize, fontname=fontn)
    setp(axs.get_xticklabels(), fontsize=tickfontsize, fontname=fontn, rotation = 45)
    setp(axs.get_yticklabels(), fontsize=tickfontsize, fontname=fontn)
    legend(loc="center left", bbox_to_anchor=(1.1, 0.5), fontsize= tickfontsize-1, frameon=false)
end

function saveplot(fname, fnamestr)
    if fname != nothing
        savefig(fname*fnamestr, bbox_inches = "tight")
    end   
end


function compute_and_plot(data, datadict, θdata, θdatadict;  
        use_data_initpt = false, stat = "mean", fname = nothing)
    
    ntrials = datadict["ntrials"]
    gamma, ugam  = get_gamma(data)
    
    pR, RT, hit = unpack_data(data, datadict; gethist = false)
    θpR, θRT, θhit = unpack_data(θdata, θdatadict; gethist = false)
    
    # pscyhometrics
    fracR = get_psychometric(pR, gamma)
    θfracR = get_psychometric(θpR, gamma)
    
    # meanRTs
    meanRT, hitRT, errRT = get_meanRTs(RT, hit, gamma, stat = stat)
    θmeanRT, θhitRT, θerrRT = get_meanRTs(θRT, θhit, gamma, stat = stat)
   
    # post error choice and RT
    post_choice = get_postsummary(pR, RT, hit, pR, gamma, stat = "mean")
    post_RT = get_postsummary(pR, RT, hit, RT, gamma, stat = stat)
    
    if use_data_initpt
        θpost_choice = get_postsummary(pR, θRT, hit, θpR, gamma, stat = "mean")
        θpost_RT = get_postsummary(pR, θRT, hit, θRT, gamma, stat = stat)
    else
        θpost_choice = get_postsummary(θpR, θRT, θhit, θpR, gamma, stat = "mean")
        θpost_RT = get_postsummary(θpR, θRT, θhit, θRT, gamma, stat = stat)
    end
    
    yy = 4.5
    xx = 4
    msize = 9
    lw = 2.5
    labpostRc = "post Right correct"
    labpostLc = "post Left correct"
    labpostRe = "post Right error"
    labpostLe = "post Left error"
    
    # PLOTTING   
    al = 0.2
    figure(figsize=(yy,xx))
    plot(ugam, fracR[:,1], color="black", label="data")
    errorbar(ugam, fracR[:,1], yerr=get_errorbars(fracR), color = "black",markersize= msize, marker ="o", linewidth = lw) 
    fill_between(ugam, θfracR[:,2], θfracR[:,3], alpha=al, color = "black")
    ylim(0.,1.)
    set_plot(gca(), ugam)
    saveplot(fname, "_main.png")
   
    figure(figsize=(yy,xx))
    plot(ugam, hitRT[:,1], color="green", label="hits")
    plot(ugam, errRT[:,1], color="red", label="errors")
    errorbar(ugam, hitRT[:,1], yerr = get_errorbars(hitRT), color="green", 
            markersize= msize, marker ="o", linewidth = lw) 
    errorbar(ugam, errRT[:,1], yerr = get_errorbars(errRT), color="red", 
            markersize= msize, marker ="o", linewidth = lw) 
    fill_between(ugam, θhitRT[:,2], θhitRT[:,3], alpha=al, color = "green")
    fill_between(ugam, θerrRT[:,2], θerrRT[:,3], alpha=al, color = "red")
    ylim(round(minimum(errRT[:,1])-0.05, digits = 2), round(maximum(hitRT[:,1])+0.05, digits = 2))
    set_plot(gca(), ugam, ylab = "Mean RT [s]")
    saveplot(fname, "_mainRT.png")
    
      
    c_postright = vec([0 164 204]./255);
    c_postleft = vec([233 115 141]./255);
    figure(figsize=(yy,xx))
    yerr_Rcorr = get_errorbars(post_choice["postRcorr"])
    yerr_Lcorr = get_errorbars(post_choice["postLcorr"])
    plot(ugam, post_choice["postRcorr"][:,1], color=c_postright)
    plot(ugam, post_choice["postLcorr"][:,1], color=c_postleft)
    errorbar(ugam, post_choice["postRcorr"][:,1], yerr = yerr_Rcorr, linewidth = lw,
        marker=">", markersize = msize, color=c_postright, label= labpostRc)
    errorbar(ugam, post_choice["postLcorr"][:,1], yerr = yerr_Lcorr, linewidth = lw,
        marker="<", markersize = msize, color=c_postleft,  label= labpostLc)
    fill_between(ugam, θpost_choice["postRcorr"][:,2], θpost_choice["postRcorr"][:,3], alpha=al, color = c_postright)
    fill_between(ugam, θpost_choice["postLcorr"][:,2], θpost_choice["postLcorr"][:,3], alpha=al, color = c_postleft)
    ylim(0.,1.)
    set_plot(gca(), ugam)
    saveplot(fname, "_postcorr.png")

 
    figure(figsize=(yy,xx))
    yerr_Rerr = get_errorbars(post_choice["postRerr"])
    yerr_Lerr = get_errorbars(post_choice["postLerr"])
    plot(ugam, post_choice["postRerr"][:,1], color=c_postright)
    plot(ugam, post_choice["postLerr"][:,1], color="orange")
    errorbar(ugam, post_choice["postRerr"][:,1], yerr = yerr_Rerr,linewidth = lw,
        marker=">", markersize = msize, color=c_postright, label= labpostRe)
    errorbar(ugam, post_choice["postLerr"][:,1], yerr = yerr_Lerr,linewidth = lw, 
        marker="<", markersize = msize, color=c_postleft, label= labpostLe)
    fill_between(ugam, θpost_choice["postRerr"][:,2], θpost_choice["postRerr"][:,3], alpha=al, color = c_postright)
    fill_between(ugam, θpost_choice["postLerr"][:,2], θpost_choice["postLerr"][:,3], alpha=al, color = c_postleft)
    ylim(0.,1.)
    set_plot(gca(), ugam)
    saveplot(fname, "_posterr.png")


    figure(figsize=(yy,xx))
    yerr_Rcorr = get_errorbars(post_RT["postRcorr"])
    yerr_Lcorr = get_errorbars(post_RT["postLcorr"])
    plot(ugam, post_RT["postRcorr"][:,1], color=c_postright)
    plot(ugam, post_RT["postLcorr"][:,1], color="orange")
    errorbar(ugam, post_RT["postRcorr"][:,1], yerr = yerr_Rcorr, linewidth = lw,
                marker=">", markersize = msize, color=c_postright, label= labpostRc)
    errorbar(ugam, post_RT["postLcorr"][:,1], yerr = yerr_Lcorr, linewidth = lw,
                marker="<", markersize = msize, color=c_postleft,  label= labpostLc)
    fill_between(ugam, θpost_RT["postRcorr"][:,2], θpost_RT["postRcorr"][:,3], alpha=al, color = c_postright)
    fill_between(ugam, θpost_RT["postLcorr"][:,2], θpost_RT["postLcorr"][:,3], alpha=al, color = c_postleft)
    ylim(minimum(errRT[:,1])-0.05, maximum(hitRT[:,1])+0.05)
    set_plot(gca(), ugam, ylab = "Mean RT [s]")
    saveplot(fname, "_postcorrRT.png")

    
    
    figure(figsize=(yy,xx))
    yerr_Rerr = get_errorbars(post_RT["postRerr"])
    yerr_Lerr = get_errorbars(post_RT["postLerr"])
    plot(ugam, post_RT["postRerr"][:,1], color=c_postright)
    plot(ugam, post_RT["postLerr"][:,1], color="orange")
    errorbar(ugam, post_RT["postRerr"][:,1], yerr = yerr_Rerr, linewidth = lw,
                marker=">", markersize = msize, color=c_postright, label= labpostRe)
    errorbar(ugam, post_RT["postLerr"][:,1], yerr = yerr_Lerr, linewidth = lw,
                marker="<", markersize = msize, color=c_postleft,  label= labpostLe)
    fill_between(ugam, θpost_RT["postRerr"][:,2], θpost_RT["postRerr"][:,3], alpha=al, color = c_postright)
    fill_between(ugam, θpost_RT["postLerr"][:,2], θpost_RT["postLerr"][:,3], alpha=al, color = c_postleft)
    ylim(minimum(errRT[:,1])-0.05, maximum(hitRT[:,1])+0.05)
    set_plot(gca(), ugam, ylab = "Mean RT [s]")
    saveplot(fname, "_posterrRT.png")
    
    figure(figsize=(yy,xx))
    bins_list = 0:0.005:maximum(RT)
    hist(RT, bins_list, density=true, alpha = 0.5, label = "data")
    hist(θRT, bins_list, density=true, alpha = 0.5, label = "fits")
    xlabel("Reaction time [s]")
    xlim(0, 2)
    legend() 
    saveplot(fname, "_hist.png")
end