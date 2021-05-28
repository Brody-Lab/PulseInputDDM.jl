"""
    A script intended to be called by a bash script for fitting the drift-diffusion linear model to one dataset.

INPUT

-folderpath: the full path of the folder containing the dataset file
-pattern: a String specifying the pattern in a file name for the file to be considered to contain data
-filenumber: specifies which file to load among the files in 'folderpath' contains specified 'pattern'

"""

using pulse_input_DDM

folderpath = ARGS[1]
pattern = ARGS[2]
filenumber = parse(Int64,ARGS[3])

filenames = readdir(folderpath);
filepaths = readdir(folderpath; joint=true)

hasdata = occursin.(filenames, pattern)
filepaths = filepaths[hasdata]

println("Loading data")
model = load_DDLM(filepaths[filenumber])
println("Loaded data. Beginning to optimize the model.")
model = optimize(model)
println("Optimized model. Beginning to save the results.")
save(model)
println("Results saved")
