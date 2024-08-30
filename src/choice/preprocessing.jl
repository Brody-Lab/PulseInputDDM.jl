"""
    bin_clicks(clicks::Vector{T})

Wrapper to broadcast bin_clicks across a vector of clicks.
"""
bin_clicks(clicks::Vector{T}; dt::Float64=1e-2, centered::Bool=false) where T <: Any =
    bin_clicks.(clicks; dt=dt, centered=centered)


"""
    bin_clicks(clicks)

Bins clicks, based on dt (defaults to 1e-2). 'centered' determines if the bin edges
occur at 0 and dt (and then ever dt after that), or at -dt/2 and dt/2 (and then
every dt after that). If the former, the bins align with the binning of spikes
in the neural model. For choice model, the latter is fine.
"""
function bin_clicks(clicks::clicks; dt::Float64=1e-2, centered::Bool=false)

    @unpack T,L,R = clicks
    nT = ceil(Int, round((T/dt), digits=10))

    if centered
        nL = searchsortedlast.(Ref((0. -dt/2):dt:(nT -dt/2)*dt), L)
        nR = searchsortedlast.(Ref((0. -dt/2):dt:(nT -dt/2)*dt), R)

    else
        nL = searchsortedlast.(Ref(0.:dt:nT*dt), L)
        nR = searchsortedlast.(Ref(0.:dt:nT*dt), R)

    end

    binned_clicks(nT, nL, nR)

end
