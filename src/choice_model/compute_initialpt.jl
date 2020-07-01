"""
exponential filter 2 params 
assumes a single process (stimulus space)

"""

function compute_initial_pt(eta::TT,beta::TT,click_data, sessbnd) where {TT <: Any}
    
    ΔLR = diffLR.(click_data)
    correct = map(ΔLR->sign(ΔLR),ΔLR)
    
    i_0 = Array{TT}(undef, length(correct))
    i_0[1] = 0.
    
    for i = 2:length(correct)
        if sessbnd[i] == 1
            i_0[i] = 0.
        else
            i_0[i] = eta*correct[i-1] + beta*i_0[i-1]
        end
    end

    return i_0 

end



"""
exponential filter 4 params 
assumes independent discounting and updating of correct and error trials (stimulus space)

"""
function compute_initial_pt(etaC::TT,etaE::TT,betaC::TT,betaE::TT, click_data, choice, sessbnd) where {TT <: Any}
    
    ΔLR = diffLR.(click_data)
    correct = map(ΔLR->ΔLR>0,ΔLR)
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
            corr = hits[rel].*etaC.*betaC.^reverse(0:length(rel)-1)
            err =  -1 .*(1 .- hits[rel]).*etaE.*betaE.^reverse(0:length(rel)-1)
            i_0[i] = sum(cho .* (corr + err))
        end
    end
   
    return  i_0

end


# function compute_initial_pt(hC::TT,eta::TT,beta::TT,click_data, sessbnd) where {TT <: Any}
    
#     ΔLR = diffLR.(click_data)
#     correct = map(ΔLR->ΔLR>0,ΔLR)
    
#     i_0 = Array{TT}(undef, length(correct))
#     i_0[1] = hC + (eta*beta/(1. - beta))/2.
    
#     for i = 2:length(correct)
#         if sessbnd[i] == 1
#             i_0[i] = hC + (eta*beta/(1. - beta))/2.
#         else
#             i_0[i] = hC*(1. - beta) + eta*beta*correct[i-1] + beta*i_0[i-1]
#         end
#     end

#     return  log.(i_0 ./(1 .- i_0))


#     ## OPTIMAL APPROXIMATION #$    

#     # ΔLR = diffLR.(click_data)
#     # correct = map(ΔLR->ΔLR>0,ΔLR)
    
#     # η_hat = 1/beta
#     # β_hat = (ibias*beta)/(1+beta)
#     # C = (1-ibias)*eta/(1-β_hat)

#     # i_0 = Array{TT}(undef, length(correct))
#     # i_0[1] = eta;
    
#     # for i = 2:length(correct)
#     #     i_0[i] = C*(1-β_hat) + η_hat*β_hat*correct[i-1] + β_hat*i_0[i-1]
#     # end

#     # return log.(i_0 ./(1 .- i_0))
# end