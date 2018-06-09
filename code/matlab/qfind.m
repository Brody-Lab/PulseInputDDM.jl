function ys=qfind(x,ts)
% function y=qfind(x,ts)
% x is a vector , t is the target (can be one or many targets),
% y is same length as ts
% does a binary search: assumes that x is sorted low to high and unique.
ys=zeros(size(ts));

for i=1:length(ts)
    t=ts(i);
    if isnan(t)
        y=nan;
    else
    high = length(x);
    low = -1;
    if t>=x(end)
        y=length(x);
    else
        try
            while (high - low > 1) 
                probe = ceil((high + low) / 2);
                if (x(probe) > t)
                    high = probe;
                else
                    low = probe;
                end
            end
            
            y=low;
        catch 
            y=low;
        end
    end
    end
    
    ys(i)=y;
end