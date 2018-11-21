function cta = brody_CTA(data)

global N dt

tri = numel(data);

Lcta = []; Rcta = [];

for i = 1:tri
    
    %loop over left clicks for this trial
    for j = 1:numel(data(i).hereL)
        %add a new entry for this click
        Lcta = cat(3,Lcta,NaN(N,0.5/dt));
        %start of firing rate
        s = data(i).hereL(j);
        %end of firing rate
        e = s + 0.5/dt - 1;
        %go to end of trial if trial ends
        e = min([data(i).nT,e]);
        %collect spike counts during that window
        Lcta(data(i).N,1:numel(s:e),end) = data(i).spike_counts(s:e,:)';
    end
    
    %do same for right clicks
    for j = 1:numel(data(i).hereR)
        Rcta = cat(3,Rcta,NaN(N,0.5/dt));
        s = data(i).hereR(j);
        e = s + 0.5/dt - 1;
        e = min([data(i).nT,e]);
        Rcta(data(i).N,1:numel(s:e),end) = data(i).spike_counts(s:e,:)';
    end
    
end

b = (1/3)* ones(1,3);
%cat over left and right, and compute mean across clicks
cta = filter(b,1,nanmean(cat(3,Rcta,-Lcta),3)'/dt)';
