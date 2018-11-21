function ccorr = compute_average_autcorr(psth,dt)

mlags = 200e-3/dt;
ccorr = NaN(size(psth,1),size(psth,1),2*mlags+1,size(psth,3));

%loop over trials
for i = 1:size(psth,3)
    %loop over neurons
    for j = 1:size(psth,1)
        
        if ~isempty(psth(j,~isnan(psth(j,:,i)),i))
            
            %loop over neurons
            for k = 1:size(psth,1)
                
                if ~isempty(psth(k,~isnan(psth(k,:,i)),i))
                    [temp,lags] = xcorr(psth(j,~isnan(psth(j,:,i)),i),...
                        psth(k,~isnan(psth(k,:,i)),i),mlags,'unbiased');
                    ts = lags(1) + mlags + 1;
                    te = lags(end) + mlags + 1;
                    ccorr(k,j,ts:te,i) = temp;
                end
                
            end
            
        end
        
    end
    
end

end
