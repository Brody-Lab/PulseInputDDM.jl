
normtanh(x) = 0.5 * (1. + tanh(x))
normatanh(x) = atanh(2. * x - 1.)

function qfind(x,ts)

    # function y=qfind(x,ts)
    # x is a vector , t is the target (can be one or many targets),
    # y is same length as ts
    # does a binary search: assumes that x is sorted low to high and unique.

    ys = zeros(Int,size(ts));

    for i = 1:length(ts)

        t = ts[i];

        if isnan(t)
            y = NaN;
        else

            high = length(x)::Int;
            low = -1;

            if t >= x[end]
                y = length(x)::Int;
            else

                try
                    while (high - low > 1)

                        probe = Int(ceil((high + low) / 2));

                        if x[probe] > t
                            high = probe;
                        else
                            low = probe;
                        end

                    end
                    
                    y = low;

                catch

                    y = low;

                end
            end
        end
        
        ys[i] = y;

    end

    return ys

end

function my_qcut(y,nconds)

    qvec = nquantile(y,nconds)
    qidx = map(x->findfirst(x .<= qvec),y)
    qidx[qidx .== 1] .= 2
    qidx .= qidx .- 2

end