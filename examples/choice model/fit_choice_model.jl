# # Loading data and fitting a choice model

# The package expects your data to live in a single .mat file which should contain a struct called `rawdata`. Each element of `rawdata` should have data for one behavioral trial and `rawdata` should contain the following fields with the specified structure:

# - `rawdata.leftbups`: row-vector containing the relative timing, in seconds, of left clicks on an individual trial. 0 seconds is the start of the click stimulus.
# - `rawdata.rightbups`: row-vector containing the relative timing in seconds (origin at 0 sec) of right clicks on an individual trial. 0 seconds is the start of the click stimulus.
# - `rawdata.T`: the duration of the trial, in seconds. The beginning of a trial is defined as the start of the click stimulus. The end of a trial is defined based on the behavioral event “cpoke_end”. This was the Hanks convention.
# - `rawdata.pokedR`: `Bool` representing the animal choice (1 = right).

# ## Fitting the model

# Once your data is correctly formatted and you have the package added in julia, you are ready to fit the model. You need to write a slurm script to use spock's resources and a .jl file to load the data and fit the model. See examples of each below. These files are also located in the package in the `examples` directory.

# ### Example slurm script

# This will start a job called `fit_choice_model`. Output will be written to a log file called `fit_choice_model.out`. This will run on the `Brody` partition of spock for 12 hours, using 44 cores and 64 GB of memory. You'll notice that we load the julia module (like we did when we added the package) and then we call julia (`-p 44` uses the 44 cores) and ask it to run the .jl file.

# ```
# #!/usr/bin/env bash
#
# #SBATCH -J 'fit_choice_model'
# #SBATCH -o ../logs/fit_choice_model.out
# #SBATCH -p Brody
# #SBATCH --time=12:00:00
# #SBATCH --mem=64000
# #SBATCH -c 44
#
# module load julia/1.0.0
# julia -p 44 ./fit_choice_model.jl
# ```

# ### Example .jl file
# Blah blah blah
# use the resources of the package
using pulse_input_DDM
using Random

# println("using ", nprocs(), " cores")

# define useful paths to the data and to a directory to save results
data_path, save_path = "data/", "results/"

# if the directory that you are saving to doesn't exist, make it
isdir(save_path) ? nothing : mkpath(save_path)

# read the name of the file located in the data directory
#files = readdir(data_path)
#files = files[.!isdir.(files)]
#file = files[1]

x = filter(x->occursin(r"^chrono_.*\d_rawdata[.]mat",x), readdir(data_path))
filenum = parse(Int64,ARGS[1])
file = x[filenum]
println(ARGS[1])

# load your data
data = load(data_path*file)

n = 53;

options = choiceoptions(fit = vcat(true, true, true, true, true, true, true, true, true, false, false),
    lb = vcat([0.,-4., -5.,-1, .5, -5., 0., 0., 0.01, 0.005], [-30, 0.]),
    ub = vcat([eps(), 4., 5., 1., 30., 5., 100., 2.5, 1.2, 1.], [30, 1.]),
    x0 = vcat([0., 0., 2., 0.8, 15., -0.5, 2., 1.5, 0.8, 0.008], [0.,1e-4]))

#if you've already ran the optimization once and want to restart from where you stoped, this will reload those parameters
# if isfile(save_file)
#     options.x0 = reload(save_file)
# end

# run the optimization
model, = optimize(data, options, n)

# compute the Hessian around the ML solution, and confidence intervals
H = Hessian(model, n)
CI, HPSD = CIs(H);

# save results
filename_save = file[1:end-4]*"_"*randstring(2)*".mat"
save_optimization_parameters(save_path*filename_save, model, options, CI)
