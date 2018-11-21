clear blah
for i = 1:length(data)
blah(i) = sum(diff([newdata.hereR{i},data(i).hereR'],1,2));
end
sum(blah)

%%

clear blah
for i = 1:length(data)
blah(i) = sum(diff([newdata.hereL{i},data(i).hereL'],1,2));
end
sum(blah)

%%

clear blah
for i = 1:length(data)
for j = 1:size(data(i).spike_counts,2)
blah(i,j) = sum(diff([newdata.spike_counts{i}(:,j),data(i).spike_counts(:,j)],1,2));
end
end

sum(sum(blah))

%%

clear blah
for i = 1:length(data)
blah(i) = sum(diff([newdata.N{i},data(i).N],1,2));
end
sum(blah)

%%

blah = sum(diff([newdata.T,cell2mat({data.T}')],1,2))
blah = sum(diff([newdata.nT,cell2mat({data.nT}')],1,2))
blah = sum(diff([newdata.pokedR,cell2mat({data.pokedR}')],1,2))

%%

clear blah
for i = 1:length(data)
blah(i) = sum(diff([newdata.rightbups{i},data(i).rightbups'],1,2));
end
sum(blah)

%%

clear blah
for i = 1:length(data)
blah(i) = sum(diff([newdata.leftbups{i},data(i).leftbups'],1,2));
end
sum(blah)

