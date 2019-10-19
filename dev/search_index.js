var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#pulse-input-DDM-1",
    "page": "Home",
    "title": "pulse input DDM",
    "category": "section",
    "text": "This is a package for inferring the parameters of drift diffusion models (DDMs) using gradient descent from neural activity or behavioral data collected when a subject is performing a pulse-based input evidence accumlation task."
},

{
    "location": "#Getting-Started-1",
    "page": "Home",
    "title": "Getting Started",
    "category": "section",
    "text": "If you want to fit the model to some data but want to interact with Julia, I recommend using a Jupyter notebook on scotty via a SSH tunnel. This is somewhat involved to set up but worth the time. See Preliminaries to fitting the model interactively on scotty for the necessary steps.To install the package, see Getting the pulse input DDM package from GitHub.If you want to fit a model non-interactively using spock, see Using spock.Other useful tips can be found here (DUO authentication required)."
},

{
    "location": "#Fitting-models-1",
    "page": "Home",
    "title": "Fitting models",
    "category": "section",
    "text": "The basic functions you need to optimize the model parameters for choice data are described in Fitting a model to choices.  If you want to optimize the model parameters for neural data, look here Fitting a model to neural activity. Pages = [\n    \"man/using_spock.md\",\n    \"man/choice_observation_model.md\",\n    \"man/neural_observation_model.md\"]\nDepth = 2"
},

{
    "location": "man/getting_the_package/#",
    "page": "Getting the pulse input DDM package from GitHub",
    "title": "Getting the pulse input DDM package from GitHub",
    "category": "page",
    "text": ""
},

{
    "location": "man/getting_the_package/#Getting-the-pulse-input-DDM-package-from-GitHub-1",
    "page": "Getting the pulse input DDM package from GitHub",
    "title": "Getting the pulse input DDM package from GitHub",
    "category": "section",
    "text": "Now you need to add the pulseinputDDM package from the github repository. Startup julia by loading the julia module on scotty or spock:    >> module load julia/1.0.0\n    >> juliathen add the package in julia by entering the pkg mode (by typing ])    (v1.0) pkg > add https://github.com/PrincetonUniversity/pulse_input_DDM/Another way to add the package (without typing ]) is to do the following, in the normal julia mode:    julia > using Pkg    \n    julia > Pkg.add(PackageSpec(url=\"https://github.com/PrincetonUniversity/pulse_input_DDM/\"))In either case, you will be prompted for your github username and password. This will require that you are part of the Princeton University github organization and the Brody Lab team. If you are not, fill out this form to get added and make sure your mention that you want to be added to the Brody Lab team.You will also need the MAT package for loading and saving MATLAB (ew) files, which you can add by typing ] add MAT."
},

{
    "location": "man/choice_observation_model-v2/#",
    "page": "Fitting the model on spock using choice data",
    "title": "Fitting the model on spock using choice data",
    "category": "page",
    "text": ""
},

{
    "location": "man/choice_observation_model-v2/#Fitting-the-model-on-spock-using-choice-data-1",
    "page": "Fitting the model on spock using choice data",
    "title": "Fitting the model on spock using choice data",
    "category": "section",
    "text": "OK, let\'s fit the model using the animal\'s choices!"
},

{
    "location": "man/choice_observation_model-v2/#Data-(on-disk)-conventions-1",
    "page": "Fitting the model on spock using choice data",
    "title": "Data (on disk) conventions",
    "category": "section",
    "text": "name_sess.mat should contain a single structure array called rawdata. Each element of rawdata should have data for one behavioral trials and rawdata should contain the following fields with the specified structure:rawdata.leftbups: row-vector containing the relative timing, in seconds, of left clicks on an individual trial. 0 seconds is the start of the click stimulus.\nrawdata.rightbups: row-vector containing the relative timing in seconds (origin at 0 sec) of right clicks on an individual trial. 0 seconds is the start of the click stimulus. \nrawdata.T: the duration of the trial, in seconds. The beginning of a trial is defined as the start of the click stimulus. The end of a trial is defined based on the behavioral event “cpoke_end”. This was the Hanks convention.\nrawdata.pokedR: Bool representing the animal choice (1 = R).\nrawdata.correct_dir: Bool representing the correct choice (1 = R). Based on the difference in left and right clicks on that trial (not the generative gamma for that trial)."
},

{
    "location": "man/choice_observation_model-v2/#Load-the-data-and-fit-the-model-interactively-1",
    "page": "Fitting the model on spock using choice data",
    "title": "Load the data and fit the model interactively",
    "category": "section",
    "text": ""
},

{
    "location": "man/choice_observation_model-v2/#Example-slurm-script-1",
    "page": "Fitting the model on spock using choice data",
    "title": "Example slurm script",
    "category": "section",
    "text": "s = \"Python syntax highlighting\"\nprint s"
},

{
    "location": "man/choice_observation_model-v2/#Exampe-.jl-file-1",
    "page": "Fitting the model on spock using choice data",
    "title": "Exampe .jl file",
    "category": "section",
    "text": "s = \"Python syntax highlighting\"\nprint s"
},

{
    "location": "man/choice_observation_model-v2/#Now-fit-the-model!-1",
    "page": "Fitting the model on spock using choice data",
    "title": "Now fit the model!",
    "category": "section",
    "text": "You can use the function optimize_model() to run the model.    pz, pd, = optimize_model(pz, pd, data)\n"
},

{
    "location": "man/neural_observation_model/#",
    "page": "Fitting a model to neural activity",
    "title": "Fitting a model to neural activity",
    "category": "page",
    "text": ""
},

{
    "location": "man/neural_observation_model/#Fitting-a-model-to-neural-activity-1",
    "page": "Fitting a model to neural activity",
    "title": "Fitting a model to neural activity",
    "category": "section",
    "text": "Now, let\'s try fiting the model using neural data."
},

{
    "location": "man/neural_observation_model/#Data-conventions-1",
    "page": "Fitting a model to neural activity",
    "title": "Data conventions",
    "category": "section",
    "text": "Following the same conventions as (Working interactively on scotty via a SSH tunnel) but rawdata the following two fields:rawdata.St: cell array containing the spike times of each neuron on an individual trial. The cell array will be length of the number of neurons recorded on that trial. Each entry of the cell array is a column vector containing the relative timing of spikes, in seconds. 0 seconds is the start of the click stimulus and spikes are retained up to the end of the trial (as defined in the field “T”).\nrawdata.cell: cell array indicating the “cellid” (as defined by Hanks convention) number for each neuron on that trial. Will be length of the number of neurons recorded on that trial. Helpful to keep track of which neuron is which, especially when multiple sessions are stitched together (not so important in the case we are discussing, of only 1 session)."
},

{
    "location": "man/neural_observation_model/#Load-the-data-and-fit-the-model-interactively-1",
    "page": "Fitting a model to neural activity",
    "title": "Load the data and fit the model interactively",
    "category": "section",
    "text": "Working from the notebook you started in the previous section (Working interactively on scotty via a SSH tunnel), we need to create three variables to point to the data we want to fit and specify which animals and sessions we want to use:data_path: a String indicating the directory where the .mat files described above are located. For example, data_path = ENV[\"HOME\"]*\"/Projects/pulse_input_DDM.jl/data\" where ENV[\"HOME\"] is using a bash environment variable and * conjoins two strings (like strjoin in MATLAB).\nratnames: A one-dimensional array of strings, where each entry is one that you want to use data from. For example, ratnames = [\"B068\",\"T034\"].\nsessids: A one-dimensional array of one-dimensional arrays of strings (get that!?) The \"outer\" 1D array should be the length of ratnames (thus each entry corresponds to each rat) and each \"inner\" array corresponds to the sessions you want to include for each rat. For example,         sessids = vcat([[46331,46484,46630]], [[151801,152157,152272]])uses sessions 46331, 46484 and 46630 from rat B068 and sessions 151801, 152157 and 152272 from rat T034."
},

{
    "location": "man/neural_observation_model/#Now-fit-the-model!-1",
    "page": "Fitting a model to neural activity",
    "title": "Now fit the model!",
    "category": "section",
    "text": "You can use the function load_and_optimize() to run the model.    pz, py = load_and_optimize(data_path,sessids,ratnames)Finally, we can save the results    using JLD\n    @save save_path*\"/results.jld\" pz pywhere save_path is specified by you."
},

{
    "location": "man/neural_observation_model/#Important-functions-1",
    "page": "Fitting a model to neural activity",
    "title": "Important functions",
    "category": "section",
    "text": "    optimize_model(pz::Vector{TT},py::Vector{Vector{TT}},pz_fit,py_fit,data;\n        dt::Float64=1e-2, n::Int=53, f_str=\"softplus\",map_str::String=\"exp\",\n        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),\n        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),\n        x_tol::Float64=1e-16,f_tol::Float64=1e-16,g_tol::Float64=1e-12,\n        iterations::Int=Int(5e3),show_trace::Bool=true, \n        λ0::Vector{Vector{Float64}}=Vector{Vector{Float64}}()) where {TT <: Any}"
},

{
    "location": "man/neural_observation_model/#Fitting-the-model-on-spock-instead-1",
    "page": "Fitting a model to neural activity",
    "title": "Fitting the model on spock instead",
    "category": "section",
    "text": "See (Using spock)."
},

{
    "location": "man/vpn_is_annoying/#",
    "page": "Efficient Remote SSH",
    "title": "Efficient Remote SSH",
    "category": "page",
    "text": ""
},

{
    "location": "man/vpn_is_annoying/#Efficient-Remote-SSH-1",
    "page": "Efficient Remote SSH",
    "title": "Efficient Remote SSH",
    "category": "section",
    "text": "(Thanks to Nick Roy of the PillowLab for these instructions!)This page is all about how to SSH into spock, della, scotty or any other campus resource without needing to use VPN.The basic idea here is to use RSA to SSH through an intermediate, open, server. In this case, the two options are nobel (nobel.princeton.edu) and arizona (arizona.princeton.edu). This should work on any unix machine (linux and MAC). Windows people should seek immediate attention. The steps are as follows:Edit (or create and edit) an ssh config on your local machine (usually located at ~/.ssh/config) using VIM or your favorite text editor. Add the following code:   Host nobel\n   User [username]\n   HostName nobel.princeton.edu\n   \n   Host spock\n   User [username]\n   HostName scotty.princeton.edu\n   ForwardX11 yes\n   ForwardX11Trusted yes\n   ProxyCommand ssh nobel -oClearAllForwardings=yes -W %h:%p   In this code, you should replace [username] with your username (i.e. what you log into each server under) and nobel can be replaced everywhere with arizona if you would like to use arizona as the pass-through server. To access other machines, replace spock with della or scotty.Create RSA keys to facilitate no-password logins more information here. The steps here are to make the RSA key:    >> ssh-keygen -t rsaThen hit enter twice to save the RSA key to the default location and to not include an RSA password. Now add the key to the pass-through server and the remote machine via:    >> ssh-copy-id [username]@nobel.princeton.edu\n    >> ssh-copy-id [username]@spock.princeton.eduwhere again [username] is your login name and you can change the address (spock.princeton.edu) to whichever machine you are trying to access. Make sure to do this first for either arizona or nobel (whichever you decide to use) and then again for the machine you are trying to access.With the above code, you can now simply ssh in via    >> ssh scotty   and not even have to worry about VPN, passwords etc.These instructions can also be found here."
},

{
    "location": "man/development/#",
    "page": "Development",
    "title": "Development",
    "category": "page",
    "text": ""
},

{
    "location": "man/development/#Development-1",
    "page": "Development",
    "title": "Development",
    "category": "section",
    "text": ""
},

{
    "location": "man/development/#Developing-the-code-1",
    "page": "Development",
    "title": "Developing the code",
    "category": "section",
    "text": ""
},

{
    "location": "man/development/#Developing-the-documents-1",
    "page": "Development",
    "title": "Developing the documents",
    "category": "section",
    "text": ""
},

{
    "location": "links/#",
    "page": "Index",
    "title": "Index",
    "category": "page",
    "text": ""
},

{
    "location": "links/#Index-1",
    "page": "Index",
    "title": "Index",
    "category": "section",
    "text": "Order   = [:type, :function]"
},

{
    "location": "functions/#",
    "page": "Functions",
    "title": "Functions",
    "category": "page",
    "text": ""
},

{
    "location": "functions/#pulse_input_DDM.compute_LL-Union{Tuple{T}, Tuple{Array{T,1},Array{T,1},Any}} where T",
    "page": "Functions",
    "title": "pulse_input_DDM.compute_LL",
    "category": "method",
    "text": "compute_LL(pz::Vector{T}, pd::Vector{T}, data; n::Int=53) where {T <: Any}\n\ncompute LL for your model. returns a scalar\n\n\n\n\n\n"
},

{
    "location": "functions/#Functions-1",
    "page": "Functions",
    "title": "Functions",
    "category": "section",
    "text": "compute_LL(pz::Vector{T}, pd::Vector{T}, data; n::Int=53) where {T <: Any}"
},

]}
