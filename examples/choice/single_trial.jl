# # Loading data and fitting a choice model
using pulse_input_DDM, Flatten, MAT, Random, JLD2
import pulse_input_DDM: P_goright
num_array = Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
src_data_folder="/scratch/ejdennis/d081_rawdata/";
src_ddm_folder="/scratch/ejdennis/d081_best/";
saveloc = "/jukebox/brody/ejdennis/"

data_files=readdir(src_data_folder);
best_files=readdir(src_ddm_folder);
filename=best_files[num_array];

file = matopen(string(src_ddm_folder,filename));
resultsdata=read(file);
close(file);

x0=resultsdata["ML_params"];
θ=θchoice(θz(x0[1],x0[2],x0[3],x0[4],x0[5],x0[6],x0[7]),x0[8],x0[9])
n=53

if occursin("false",filename)
    cross=false;
else
    cross=true;
end

for i =1:length(data_files)
    if occursin(filename[1:4],data_files[i])
	data = load_choice_data(string(src_data_folder,data_files[i]));
	model = choiceDDM(θ=θ,data=data,n=n,cross=cross);
	datatosave=P_goright(model);
	filenametosave=string(saveloc,filename[1:end-4],i,"_P_goright.mat")
	file = matopen(filenametosave,"w")
   	write(file,"Pright",datatosave)
   	close(file)
    end
end

