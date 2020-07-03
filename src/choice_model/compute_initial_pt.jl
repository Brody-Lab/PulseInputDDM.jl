"""
exponential filter 2 params 
assumes a single process (stimulus space)
"""

function compute_initial_pt(hist_θz::θz_expfilter, bias::TT, click_data,
             sessbnd::Vector{Bool}, choice = 0.) where TT <: Any
    
    @unpack h_η, h_β = hist_θz
    correct = map(inputs->sign(inputs.clicks.gamma), click_data)
    i_0 = Array{TT}(undef, length(correct)) 
    i_0[1] = 0.
    
    for i = 2:length(correct)
        if sessbnd[i] == 1
            i_0[i] = 0.
        else
            i_0[i] = h_η*correct[i-1] + h_β*i_0[i-1]
        end
    end

    return i_0 

end



"""
exponential filter 4 params 
assumes independent discounting and updating of correct and error trials (stimulus space)
"""
function compute_initial_pt(hist_θz::θz_expfilter_ce, bias::TT, click_data,
             sessbnd::Vector{Bool}, choice) where TT <: Any
    
    @unpack h_ηC, h_ηE, h_βC, h_βE = hist_θz

    correct = map(inputs->sign(inputs.clicks.gamma), click_data)    
    hits = correct .== choice
    sessbnd[1] = 1
    lim = 1

    i_0 = Array{TT}(undef, length(correct))
    
    for i = 1:length(correct)
        if sessbnd[i] == 1
            lim = i
            i_0[i] = 0.
            rel = []
         else
            rel = max(lim, i-10):i-1
            cho = -1. .*(1 .- choice[rel]) + choice[rel]
            corr = hits[rel].*h_ηC.*h_βC.^reverse(0:length(rel)-1)
            err =  -1 .*(1 .- hits[rel]).*h_ηE.*h_βE.^reverse(0:length(rel)-1)
            i_0[i] = sum(cho .* (corr + err))
        end
    end
   
    return  i_0

end



#=
function compute_initial_value(data::Dict, η::TT, α_prior::TT, β_prior::TT) where {TT <: Any}

    case = 4

    # L-R DBM
    if case == 1

        correct = data["correct"]

        α_actual = α_prior * β_prior
        β_actual = β_prior - α_actual

        prior = Beta(α_actual, β_actual)
        x = collect(0.001 : 0.001 : 1. - 0.001)
        prior_0 = pdf.(prior,x)
        prior_0 = prior_0/sum(prior_0)
        
        post = Array{Float64}(undef, size(prior_0))
        cprob = Array{TT}(undef, data["ntrials"]) 

        for i = 1:data["ntrials"]
            if data["sessidx"][i] == 1
                prior_i = prior_0
                Ep_x1_xt_1 = sum(x.*prior_i)
                cprob[i] = Ep_x1_xt_1
            else
                prior_i = η*post + (1-η)*prior_0
                Ep_x1_xt_1 = sum(x.*prior_i)
           
                cprob[i] = Ep_x1_xt_1
            end

            if correct[i] == 1
                post = x.*prior_i
            else
                post = (1 .- x).* prior_i
            end
            post = post./sum(post)
        end

        return log.(cprob ./(1 .- cprob))


    # Alt-rep DBM    
    elseif case == 2

        correct = data["correct"]
        ra = abs.(diff(correct))
        ra = vcat(0, ra)

        α_actual = α_prior * β_prior
        β_actual = β_prior - α_actual

        prior = Beta(α_actual, β_actual)
        x = collect(0.001 : 0.001 : 1. - 0.001)
        prior_0 = pdf.(prior,x)
        prior_0 = prior_0/sum(prior_0)
        
        post = Array{Float64}(undef, size(prior_0))
        cprob = Array{TT}(undef, data["ntrials"]) 

        for i = 1:data["ntrials"]
            if data["sessidx"][i] == 1
                prior_i = prior_0
                Ep_x1_xt_1 = sum(x.*prior_i)
                cprob[i] = Ep_x1_xt_1
            else
                prior_i = η*post + (1-η)*prior_0
                Ep_x1_xt_1 = sum(x.*prior_i)
                if correct[i-1] == 1
                    cprob[i] = Ep_x1_xt_1
                else
                    cprob[i] = 1-Ep_x1_xt_1
                end
               
            end

            if ra[i] == 1
                post = (1 .- x).* prior_i
            else
                post = x.*prior_i
            end

           
            post = post./sum(post)
        end

        return log.(cprob ./(1 .- cprob))


    # exponential L-R    
    elseif case == 3
        correct = data["correct"]
        cprob = Array{TT}(undef, data["ntrials"]) 

        η_hat = 1/β_prior
        β_hat = (η*β_prior)/(1+β_prior)
        C     = (1-η)*α_prior/(1-β_hat)
        for i = 1:data["ntrials"]
            if data["sessidx"][i] == 1
                cprob[i] = α_prior
            else
                cprob[i] = C*(1-β_hat) + η_hat*β_hat*correct[i-1] + β_hat*cprob[i-1]
            end
        end
            
        return log.(cprob ./(1 .- cprob))



    # exponential approx L-R    
    elseif case == 4
        correct = data["correct"]
        cprob = Array{TT}(undef, data["ntrials"]) 

        for i = 1:data["ntrials"]
            if data["sessidx"][i] == 1
                cprob[i] = 0.
            else
                cprob[i] = η + α_prior*correct[i-1] + β_prior*cprob[i-1]
            end    
        end

        return cprob
        

    end


end

=#