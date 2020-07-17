"""
    exponential filter 2 params 
    assumes a single process (stimulus space)

"""

function compute_initial_pt(hist_θz::θz_expfilter, σ2_s::TT, data_dict) where TT <: Any
    
    @unpack h_η, h_β = hist_θz
    ntrials = length(data_dict["correct"])
    i_0 = Array{TT}(undef, ntrials) 
    i_0[1] = 0.
    
    for i = 2:length(ntrials)
        if data_dict["sessbnd"][i] == 1
            i_0[i] = 0.
        else
            i_0[i] = h_η*data_dict["correct"][i-1] + h_β*i_0[i-1]
        end
    end

    return i_0 

end



"""
    exponential filter 4 params 
    assumes independent discounting and updating of correct and error trials (stimulus space)

"""
function compute_initial_pt(hist_θz::θz_expfilter_ce, σ2_s::TT, data_dict) where TT <: Any
    
    @unpack h_ηC, h_ηE, h_βC, h_βE = hist_θz
    hits = data_dict["correct"] .== data_dict["choice"]
    ntrials = length(hits)
    data_dict["sessbnd"][1] = 1
    lim = 1

    i_0 = Array{TT}(undef, ntrials)
    
    for i = 1:ntrials
        if data_dict["sessbnd"][i] == 1
            lim = i
            i_0[i] = 0.
            rel = []
         else
            rel = max(lim, i-10):i-1
            cho = -1. .*(1 .- data_dict["choice"][rel]) + data_dict["choice"][rel]
            corr = hits[rel].*h_ηC.*h_βC.^reverse(0:length(rel)-1)
            err =  -1 .*(1 .- hits[rel]).*h_ηE.*h_βE.^reverse(0:length(rel)-1)
            i_0[i] = sum(cho .* (corr + err))
        end
    end
   
    return  i_0

end

"""
    exponential filter 4 params 
    assumes independent discounting and updating of correct and error trials (stimulus space)
    takes an additional last trial argument
"""
function compute_initial_pt(hist_θz::θz_expfilter_ce, σ2_s::TT, data_dict, tr::Int) where TT <: Any
    
    @unpack h_ηC, h_ηE, h_βC, h_βE = hist_θz
    hits = data_dict["correct"][1:tr] .== data_dict["choice"][1:tr]
    data_dict["sessbnd"][1] = 1
    lim = 1

    i_0 = Array{TT}(undef, tr)
    
    for i = 1:tr
        if data_dict["sessbnd"][i] == 1
            lim = i
            i_0[i] = 0.
            rel = []
         else
            rel = max(lim, i-10):i-1
            cho = -1. .*(1 .- data_dict["choice"][rel]) + data_dict["choice"][rel]
            corr = hits[rel].*h_ηC.*h_βC.^reverse(0:length(rel)-1)
            err =  -1 .*(1 .- hits[rel]).*h_ηE.*h_βE.^reverse(0:length(rel)-1)
            i_0[i] = sum(cho .* (corr + err))
        end
    end
   
    return  i_0

end


"""
"""
function compute_bnd(θ::θ_DBMexpbnd, σ2_s::TT, data_dict) where TT <: Any
    @unpack Bc, Be = θ
    @unpack B0 = θ.base_θz

    hits = data_dict["correct"] .== data_dict["choice"]
    B = Array{TT}(undef, data_dict["ntrials"])

    for i = 1:data_dict["ntrials"]
        if data_dict["sessbnd"][i] == 1
            B[i] = B0
        else
            hits[i-1] ? B[i] = B[i-1] + Bc : B[i] = B[i-1] + Be
        end
    end
end



"""
"""
function compute_bnd(θ::DDMθ, σ2_s::TT, data_dict) where TT <: Any
    @unpack B0 = θ.base_θz
    B = Array{TT}(undef, data_dict["ntrials"])
    B .= B0
end


"""
    DBM initial point: returns value in log posterior units

"""
function compute_initial_pt(hist_θz::θz_DBM, σ2_s::TT, data_dict) where TT <: Any

    @unpack h_α, h_u, h_v = hist_θz
    α_prior = h_u * h_v
    β_prior = h_v - α_prior

    prior = Beta(α_prior, β_prior)
    x = collect(0.001:0.001: 1. - 0.001)
    prior_0 = pdf.(prior, x)
    prior_0 = prior_0/sum(prior_0)

    post = Array{Float64}(undef,size(prior_0))
    cprob = Array{TT}(undef, data_dict["ntrials"])

    for i = 1:data_dict["ntrials"]
        data_dict["sessbnd"][i] ? prior_i = prior_0 : prior_i = h_α*post + (1-h_α)*prior_0
        cprob[i] = sum(x.*prior_i)
        data_dict["correct_bin"][i] ? post = x.*prior_i : post = (1. .- x).*prior_i
        post = post./sum(post)
    end

    return log.(cprob ./ (1 .- cprob))

end


"""
    DBMexp initial point: returns value in log posterior units

"""
function compute_initial_pt(hist_θz::θz_DBMexp, σ2_s::TT, data_dict) where TT <: Any

    @unpack h_α, h_u, h_v = hist_θz
    η = 1/h_v
    β = (h_α*h_v)/(1+h_v)
    C = (1-h_α)*h_u/(1-β)

    inval = C + (η*β/(2*(1-β)))   # mean value of the exponential filter
    cprob = Array{TT}(undef, data_dict["ntrials"])

    for i = 1:data_dict["ntrials"]
        data_dict["sessbnd"][i] ? cprob[i] = inval : cprob[i] = (1-β)*C + β*(η*data_dict["correct_bin"][i-1] + cprob[i-1])
    end

    return log.(cprob ./ (1 .- cprob))

end  


"""
    LPSexp initial point: returns value in log posterior units

"""
function compute_initial_pt(hist_θz::θz_LPSexp, σ2_s::TT, data_dict) where TT <: Any

    @unpack h_α, h_β, h_C = hist_θz

    inval = (2*h_C + h_α)/(2*(1-h_β))
    cprob = Array{TT}(undef, data_dict["ntrials"])

    for i = 1:data_dict["ntrials"]
        data_dict["sessbnd"][i] ? cprob[i] = inval : cprob[i] = h_C + h_α*data_dict["correct_bin"][i-1] + h_β*cprob[i-1]
    end

    return log.(cprob ./ (1 .- cprob))

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