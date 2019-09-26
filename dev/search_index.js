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
    "location": "man/setting_things_up_on_scotty/#",
    "page": "Preliminaries to fitting the model interactively on scotty",
    "title": "Preliminaries to fitting the model interactively on scotty",
    "category": "page",
    "text": ""
},

{
    "location": "man/setting_things_up_on_scotty/#Preliminaries-to-fitting-the-model-interactively-on-scotty-1",
    "page": "Preliminaries to fitting the model interactively on scotty",
    "title": "Preliminaries to fitting the model interactively on scotty",
    "category": "section",
    "text": "First you need to install Julia in your scotty user directory via anaconda. This will be a whole new installation of Julia (not the installation you might normally use on scotty if you typed module load julia) which means if you\'ve used another installation of Julia on scotty before, all the packages you might have added won\'t be there. Using a new anaconda installation of Julia allows easier use of Jupyter notebooks via a SSH tunnel. "
},

{
    "location": "man/setting_things_up_on_scotty/#Installing-Julia-via-anaconda-1",
    "page": "Preliminaries to fitting the model interactively on scotty",
    "title": "Installing Julia via anaconda",
    "category": "section",
    "text": "From your local machine, make a ssh connection with scotty,     >> ssh [username]@scotty.princeton.eduwhere [username] is your scotty username, assuming you have one. If you do not have a username email PNI help.(If you haven\'t configured your local machine so that you can avoid using the VPN to access scotty, do that first. Instructions are provided on the Efficient Remote SSH page.)Next load an anaconda module on scotty (for example, 5.1.0),     >> module load anacondapy/5.1.0Presently, you are using the \"base\" anaconda environment, for which PNI users do not have write privileges. You need to create a new environment, which we will call \"Julia\",    >> conda create --name juliaNow that you\'ve created this new environment, you need to \"activate\" it,     >> source activate juliaNext, you need to install Julia within this environment via anaconda,     >> conda install -c conda-forge juliaYou are using a whole new version of Julia (not the one you use when you execute module load julia on scotty or spock), so you will need to re-install lots of packages (if you have used Julia on scotty or spock before), most specifically, the IJulia package so you can use Julia in a Jupyter notebook."
},

{
    "location": "man/setting_things_up_on_scotty/#Installing-the-IJulia-package-1",
    "page": "Preliminaries to fitting the model interactively on scotty",
    "title": "Installing the IJulia package",
    "category": "section",
    "text": "Once, Julia is installed, you can launch it by typing julia. From here, you need to add the IJulia package so you can use Julia in a notebook. In Julia 1.0, you can access the \"package manager\" by pressing ]. From there, enter    (v1.0) pkg > add IJuliaYou might now want to add any packages you use, again in the normal way (for example, you can now add the pulse_input_DDM package, as we will do eventually, described in the Getting the pulse input DDM package from GitHub section). You\'re done (with this part)! You can exit Julia."
},

{
    "location": "man/setting_things_up_on_scotty/#Opening-a-notebook-on-scotty-and-creating-a-SSH-tunnel-1",
    "page": "Preliminaries to fitting the model interactively on scotty",
    "title": "Opening a notebook on scotty and creating a SSH tunnel",
    "category": "section",
    "text": "Now, that Julia is all set up, you need to launch a Jupyter notebook on scotty (which is possible because you\'re already using the anaconda module, which contains jupyter). In your ssh connection to scotty, type     >> jupyter notebookOnce this has executed, it will pick a port on which to run and provide you a url so that you can launch the notebook in your local browser. But before you can do that, you need to create a \"ssh tunnel\" from the open port on scotty and a port on your local machine. In a separate terminal window, create a new ssh connection to scotty, but one that will map the ports,     >> ssh -L <port>:localhost:<port> [username]@scotty.princeton.eduwhere <port> is the port assigned by scotty, and the second <port> is one on your local machine (you can pick the same one that scotty did, if you don\'t happen to be already using it).Now, copy the url that was created when you launched the jupyter notebook into your local browser, and voila! On a mac, you can press the \"command\" key (⌘), and the url should become underlined, at which point you can click on it with the  mouse and the link should open in your local browser.Here is a screen shot of what a typical terminal will look like, showing the url (at the bottom) that needs to be copied and pasted or clicked on:(Image: notebook-screen-shot)"
},

{
    "location": "man/setting_things_up_on_scotty/#Now-that-everything\'s-set-up,-how-do-I-open-a-notebook-*again*?-1",
    "page": "Preliminaries to fitting the model interactively on scotty",
    "title": "Now that everything\'s set up, how do I open a notebook again?",
    "category": "section",
    "text": "Next time, you only need to:SSH into scotty: $ ssh scotty.\nLoad the anaconda module: $ module load anacondapy/5.1.0.\nActivate the julia environment: $ source activate julia.\nLaunch a jupyter notebook: $ jupyter notebook.\nCreate the ssh tunnel: $ ssh -L <port>:localhost:<port> scotty.These instructions can also be found here."
},

{
    "location": "man/setting_things_up_on_scotty/#Running-interactively-on-spock-brody-instead-1",
    "page": "Preliminaries to fitting the model interactively on scotty",
    "title": "Running interactively on spock-brody instead",
    "category": "section",
    "text": "I\'m not sure if this is technically supported by PNI IT or not, but sometimes scotty can be bogged down with lots of jobs. Our personal computing cluster (spock-brody) that PNI manages is mostly set up for batch jobs (like spock) but can be hyjacked to run interactively by executing the following from spock (once you made a SSH connection to it):    $ salloc -p Brody -t 24:00:00 -c 44 srun --pty bashThis will request 1 node (44 cores) for 24 hours from the Brody partition. The second bit srun --pty bash will open a bash shell on the compute node you just requested. You can now use it like you used scotty above. "
},

{
    "location": "man/setting_things_up_on_scotty/#Jupyter-notebook-server-on-spock-brody-1",
    "page": "Preliminaries to fitting the model interactively on scotty",
    "title": "Jupyter notebook server on spock-brody",
    "category": "section",
    "text": "If you wanted to run a jupyter notebook server on spock-brody, you need to do some fancy SSH tunneling to get it to work. Here\'s how to do the whole thing, beginning to end.First, you need to create a file and modify some jupyter defaults. Without doing this, jupyter tries to write some files in a location that you won\'t have permissions for. Next, you need to create a \"cookie secret\" file (I have no idea what this is, but it sounds delicious!) You can find details about this here. $ openssl rand -hex 32 > [path]/jupyterhub_cookie_secret. Here [path] is something you decide and have write access to. For example I used /mnt/bucket/people/briandd/.jupyter/.\nNext, you need to create a default jupyter config file by typing $ jupyter notebook --generate-config. I learned about this here. Now you need to edit this file to indicate the location of the cookie secret you just made. Modify the lines in the newly created file jupyter_notebook_config.py, which is located in $HOME/.jupyter so that it reads    c.Notebook.cookie_secret_file = \'[path]/jupyter_cookie_secret\'OK, now we can go ahead and open our jupyter notebook and create our crazy SHH tunnel.ssh spock\ntmux new -s [name] \\; split-window -h \\; attach, where [name] is whatever you want to name the tmux session.\nIn one tmux pane, you are going request the resources you want from spock-brody and create the jupyter notebook, and then in the other, you are going to create a SSH tunnel from the spock login node to the compute node you are currently using. Then, you have to create a SSH tunnel from your local machine to the spock login node (as we have done before). \nIn detail, in one pane:\n\n  ```\n      >> salloc -p Brody -t 24:00:00 -c 44 srun --pty bash\n  ```\n      \n  which will automatically create and \"move you\" over to a bash shell in the compute node.\nmodule load anacondapy/5.1.0\nsource activate julia\nEach time you try to do this, you have to $ unset XDG_RUNTIME_DIR. I\'m not exactly sure what this does, but I discovered this solution here.\njupyter notebook\nNow you should have a jupyter notebook server running in that pane (which is actually a bash shell running on the spock-brody compute node).\nIn the other pane, you need to create a SSH tunnel from the spock login node (which is where this panel exists) and the spock-brody compute node you are using. The name of that node should have been reported back to you when you ran salloc, for example it might be called spock-brody01-01. To creat the tunnel, type ssh -L <port1>:localhost:<port2> [node-name] where <port1> is an open port on the spock login node, <port2> is the port on the spock compute node where your jupyter notebook is running and [node-name] is the name of the spock compute node you are using.\nFinally, detach you tmux session <crtl-D> and exit your SSH connection with spock. Now, create another SSH tunnel from your local machine to the spock login node ssh -L <port1>:localhost:<port2> spock where <port1> is an open port on your local machine and <port2> is the port you assigned the SSH tunnel to on the spock login node. After you do this, you will have an SSH connection again with the spock login node. To re-attach your tmux session, type tmux attach -t [name]. And then, you can clikc on the link provided by the jupyter notebook, which should open in a browse on your local machine. Phew!\nIn the second tmux pane, you can open an htop to see your activity on the compute node htop -u [username]."
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
    "text": "Now you need to add the pulseinputDDM package from the github repository. Startup up a \"anaconda julia\" REPL the same way we did above when you installed the IJulia pacakge    >> ssh scotty\n    >> module load anacondapy/5.1.0\n    >> juliathen add the package in julia by entering the pkg mode (by typing ])    (v1.0) pkg > add https://github.com/PrincetonUniversity/pulse_input_DDM/Another way to add the package (without typing ]) is to do the following, in the normal julia mode:    julia > using Pkg    \n    julia > Pkg.add(PackageSpec(url=\"https://github.com/PrincetonUniversity/pulse_input_DDM/\"))In either case, you will be prompted for your github username and password. This will require that you are part of the Princeton University github organization and the Brody Lab team. If you are not, fill out this form to get added and make sure your mention that you want to be added to the Brody Lab team."
},

{
    "location": "man/working_interactively_on_scotty/#",
    "page": "Working interactively on scotty via a SSH tunnel",
    "title": "Working interactively on scotty via a SSH tunnel",
    "category": "page",
    "text": ""
},

{
    "location": "man/working_interactively_on_scotty/#Working-interactively-on-scotty-via-a-SSH-tunnel-1",
    "page": "Working interactively on scotty via a SSH tunnel",
    "title": "Working interactively on scotty via a SSH tunnel",
    "category": "section",
    "text": "You should be all set up to create a SSH tunnel to scotty (if not check out the section Preliminaries to fitting the model interactively on scotty). Follow the steps in the section Now that everything\'s set up, how do I open a notebook again? to create a new SSH tunnel and launch a new jupyter notebook server."
},

{
    "location": "man/working_interactively_on_scotty/#Running-the-model-using-a-notebook-1",
    "page": "Working interactively on scotty via a SSH tunnel",
    "title": "Running the model using a notebook",
    "category": "section",
    "text": "OK, we\'re ready to analyze some data! Once you\'ve created your SSH tunnel, open a new notebook and call it whatever you like and save it whereever you like. Add this to the first cell of your notebook    using Distributed\n    addprocs([some number])where [some number] is the number of extra cores you want to have access to on scotty. using Distributed is necessary to use julia\'s parallel computation features. In the next cell, enter    @everywhere using pulse_input_DDMNow you have imported the package into the main namespace (which means you can use it\'s functionality) and you\'ve placed it on all of the processors by the @everywhere.OK, now move on to the section for fitting to choice data (Fitting a model to choices) or neural data (Fitting a model to neural activity). "
},

{
    "location": "man/choice_observation_model/#",
    "page": "Fitting a model to choices",
    "title": "Fitting a model to choices",
    "category": "page",
    "text": ""
},

{
    "location": "man/choice_observation_model/#Fitting-a-model-to-choices-1",
    "page": "Fitting a model to choices",
    "title": "Fitting a model to choices",
    "category": "section",
    "text": "OK, let\'s fit the model using the animal\'s choices!"
},

{
    "location": "man/choice_observation_model/#Data-(on-disk)-conventions-1",
    "page": "Fitting a model to choices",
    "title": "Data (on disk) conventions",
    "category": "section",
    "text": "If you\'re using .mat files, the package expects data to be organized in a directory in a certain way, following a certain name convention and the data in each .mat file should follow a specific convention. Data from session sess for rat name should be named name_sess.mat. All of the data you want to analyze should be located in the same directory. name_sess.mat should contain a single structure array called rawdata. Each element of rawdata should have data for one behavioral trials and rawdata should contain the following fields with the specified structure:rawdata.leftbups: row-vector containing the relative timing, in seconds, of left clicks on an individual trial. 0 seconds is the start of the click stimulus.\nrawdata.rightbups: row-vector containing the relative timing in seconds (origin at 0 sec) of right clicks on an individual trial. 0 seconds is the start of the click stimulus. \nrawdata.T: the duration of the trial, in seconds. The beginning of a trial is defined as the start of the click stimulus. The end of a trial is defined based on the behavioral event “cpoke_end”. This was the Hanks convention.\nrawdata.pokedR: Bool representing the animal choice (1 = R).\nrawdata.correct_dir: Bool representing the correct choice (1 = R). Based on the difference in left and right clicks on that trial (not the generative gamma for that trial)."
},

{
    "location": "man/choice_observation_model/#Load-the-data-and-fit-the-model-interactively-1",
    "page": "Fitting a model to choices",
    "title": "Load the data and fit the model interactively",
    "category": "section",
    "text": "Working from the notebook you started in the previous section (Working interactively on scotty via a SSH tunnel), we need to create three variables to point to the data we want to fit and specify which animals and sessions we want to use:data_path: a String indicating the directory where the .mat files described above are located. For example, data_path = ENV[\"HOME\"]*\"/Projects/pulse_input_DDM.jl/data\" where ENV[\"HOME\"] is using a bash environment variable and * conjoins two strings (like strjoin in MATLAB).\nratnames: A one-dimensional array of strings, where each entry is one that you want to use data from. For example, ratnames = [\"B068\",\"T034\"].\nsessids: A one-dimensional array of one-dimensional arrays of strings (get that!?) The \"outer\" 1D array should be the length of ratnames (thus each entry corresponds to each rat) and each \"inner\" array corresponds to the sessions you want to include for each rat. For example,         sessids = vcat([[46331,46484,46630]], [[151801,152157,152272]])uses sessions 46331, 46484 and 46630 from rat B068 and sessions 151801, 152157 and 152272 from rat T034."
},

{
    "location": "man/choice_observation_model/#Now-fit-the-model!-1",
    "page": "Fitting a model to choices",
    "title": "Now fit the model!",
    "category": "section",
    "text": "You can use the function load_and_optimize() to run the model.    pz, pd = load_and_optimize(data_path,sessids,ratnames)Finally, we can save the results    using JLD\n    @save save_path*\"/results.jld\" pz pdwhere save_path is specified by you."
},

{
    "location": "man/choice_observation_model/#Fitting-the-model-on-spock-instead-1",
    "page": "Fitting a model to choices",
    "title": "Fitting the model on spock instead",
    "category": "section",
    "text": "See (Using spock)."
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
    "location": "man/using_spock/#",
    "page": "Using spock",
    "title": "Using spock",
    "category": "page",
    "text": ""
},

{
    "location": "man/using_spock/#Using-spock-1",
    "page": "Using spock",
    "title": "Using spock",
    "category": "section",
    "text": "We can fit the parameters of the latent model uses animal choices."
},

{
    "location": "man/using_spock/#Key-functions-1",
    "page": "Using spock",
    "title": "Key functions",
    "category": "section",
    "text": ""
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
    "text": "compute_LL(pz::Vector{T}, pd::Vector{T}, data; n::Int=53)\n\ncompute LL for your model. returns a scalar\n\n\n\n\n\n"
},

{
    "location": "functions/#Functions-1",
    "page": "Functions",
    "title": "Functions",
    "category": "section",
    "text": "compute_LL(pz::Vector{T}, pd::Vector{T}, data; n::Int=53) where {T <: Any}"
},

]}
