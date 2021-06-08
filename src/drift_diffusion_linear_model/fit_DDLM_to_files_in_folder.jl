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
hasdata = occursin.(filenames, pattern)
filepaths = map(x-> joinpath(folderpath, x), filenames[hasdata])

println("======")
println("Loading data")
model = load_DDLM(filepaths[filenumber])
println("Loaded data. Beginning to optimize the model.")
model, output = optimize(model)
println("Optimized model. Beginning to save the results.")
save(model)
println("Results saved")
println("======")

using Optim
Optim.converged(output) || error("Failed to converge in $(Optim.iterations(output)) iterations")
