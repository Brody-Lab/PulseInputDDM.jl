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
    "text": "This is a package for inferring the parameters of drift diffusion models (DDMs) using gradient descent from spiking neural activity or choice data collected when a subject is performing a pulsed input evidence accumlation task."
},

{
    "location": "#Getting-Started-1",
    "page": "Home",
    "title": "Getting Started",
    "category": "section",
    "text": "If you want to fit the model to some data, but want to interact with Julia, I recommend using a Jupyter notebook and scotty. This is somewhat involved to set up, but worth the time. Here are the steps necessary to do this."
},

{
    "location": "#Fitting-the-model-interactively-on-scotty-1",
    "page": "Home",
    "title": "Fitting the model interactively on scotty",
    "category": "section",
    "text": "First you need to install Julia in your local scotty user directory via anaconda. This will be a whole new installation of Julia (not the installation you might normally use on scotty if you typed module load julia) which means if you\'ve used another installation of Julia on scotty before, all of the packages you might have added won\'t be there. The reason for using a new anaconda installation of Julia is that "
},

{
    "location": "#Installing-Julia-via-anaconda-1",
    "page": "Home",
    "title": "Installing Julia via anaconda",
    "category": "section",
    "text": "From your local machine, make a ssh connection with scotty,     >> ssh [username]@scotty.princeton.eduwhere [username] is your scotty username, assuming you have one. If you do not have a username email <pnihelp@princeton.edu>.(If you haven\'t configured your local machine so that you can avoid using the VPN to access scotty, first do that. Instructions for doing so are provided on the Efficient Remote SSH page.)Next load an anaconda module (for example, 5.1.0),     >> module load anacondapy/5.1.0Presently, you are using the \"base\" anaconda environment, for which PNI users do not have write privileges. You need to create a new environment, which we will call \"Julia\",    >> conda create --name juliaNow that you\'ve created this new environment, you need to use it, source activate julia. Next, you need to install Julia via anaconda,     >> conda install -c conda-forge juliaYou are using a whole new version of Julia (not the same one as when you execute module load julia on scotty or spock), so you will need to re-install lots of packages (if you have used Julia on scotty or spock before), most specifically, the IJulia package."
},

{
    "location": "#Installing-the-IJulia-package-1",
    "page": "Home",
    "title": "Installing the IJulia package",
    "category": "section",
    "text": "Once, Julia is installed, you can launch it by typing julia. From here, you need to add the IJulia package so you can use Julia in a notebook. This is done in the normal way,    julia > using Pkg\n    julia > Pkg.add(\"IJulia\")You might now want to add any packages you use, again in the normal way (for example, you can now add the pulseinputDDM package, as we will do eventually, described in the Getting the pulseinputDDM package from GitHub section). You\'re done (with this part)! You can exit Julia."
},

{
    "location": "#Opening-a-notebook-on-scotty-and-creating-a-SSH-tunnel-1",
    "page": "Home",
    "title": "Opening a notebook on scotty and creating a SSH tunnel",
    "category": "section",
    "text": "Now, that Julia is all set up, you need to launch a Jupyter notebook on scotty (which is possible because you\'re already using the anaconda module, which contains jupyter). In your ssh connection to scotty, type jupyter notebook. Once this has executed, it will pick a port on which to run and provide you a url so that you can launch the notebook in your local browser. But before you can do that, you need to create a \"ssh tunnel\" from the open port on scotty and a port on your local machine. In a separate terminal window, create a new ssh connection to scotty, but one that will map the ports, ssh -L <port>:localhost:<port> [username]@scotty.princeton.edu where <port> is the port assigned by scotty, and the second <port> is one on your local machine (you can pick the same one that scotty did, if you don\'t happen to be already using it).Now, copy the url that was created when you launched the jupyter notebook into your local browser, and voila! On a mac, you can press the \"command\" key (⌘), and the url should become underlined, at which point you can click on it with the  mouse and the link should open in your local browser.Here is a screen shot of what a typical terminal will look like, showing the url that needs to be copied and pasted:(Image: notebook-screen-shot)"
},

{
    "location": "#Now-that-everything\'s-set-up,-how-do-I-open-a-notebook-again?-1",
    "page": "Home",
    "title": "Now that everything\'s set up, how do I open a notebook again?",
    "category": "section",
    "text": "Next time, you only need to:SSH into scotty: ssh scotty.\nLoad the anaconda module: module load anacondapy/5.1.0.\nActivate the julia environment: source activate julia.\nLaunch a jupyter notebook: jupyter notebook.\nCreate the ssh tunnel: ssh -L <port>:localhost:<port> scotty.These instructions can also be found here."
},

{
    "location": "#Getting-the-pulse*input*DDM-package-from-GitHub-1",
    "page": "Home",
    "title": "Getting the pulseinputDDM package from GitHub",
    "category": "section",
    "text": "Now, you need to add the pulseinputDDM package from the github repository. Startup up a \"anaconda julia\" REPL the same way we did above when you installed the IJulia pacakge, then  by typing the following commands into a Julia REPL:    julia > using Pkg\n    \n    julia > Pkg.add(PackageSpec(url=\"https://github.com/PrincetonUniversity/pulse_input_DDM/\"))You will be prompted for your github username and password. This will require that you are part of the Princeton University github organization and the Brody Lab team. If you are not, fill out this form to get added and make sure your mention that you want to be added to the Brody Lab team."
},

{
    "location": "#Basics-1",
    "page": "Home",
    "title": "Basics",
    "category": "section",
    "text": "Pages = [\n    \"man/using_spock.md\",\n    \"man/aggregating_sessions.md\",\n    \"man/choice_observation_model.md\",\n    \"man/neural_observation_model.md\"]\nDepth = 2"
},

{
    "location": "#To-do-1",
    "page": "Home",
    "title": "To do",
    "category": "section",
    "text": "RBF optimization and compare\nInstructions to modify docs\nInstructions to modify package (forking, branching, git related things)\nInstructions for formatting data\nInstructions for running optimizations and looking at results\nInstructions for running on spock.\nShell scripts for running on spock."
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
    "text": "We can fit the parameters of the latent model uses animal choices."
},

{
    "location": "man/choice_observation_model/#Key-functions-1",
    "page": "Fitting a model to choices",
    "title": "Key functions",
    "category": "section",
    "text": "Here\'s some test mathfracnk(n - k) = binomnk"
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
    "text": "Some text describing this section"
},

{
    "location": "man/neural_observation_model/#pulse_input_DDM.optimize_model-Union{Tuple{TT}, Tuple{Array{TT,1},Array{Array{TT,1},1},Any,Any,Any}} where TT",
    "page": "Fitting a model to neural activity",
    "title": "pulse_input_DDM.optimize_model",
    "category": "method",
    "text": "optimize_model(pz,py,pz_fit,py_fit,data;\n    dt::Float64=1e-2, n::Int=53, f_str=\"softplus\",map_str::String=\"exp\",\n    beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),\n    mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),\n    x_tol::Float64=1e-16,f_tol::Float64=1e-16,g_tol::Float64=1e-12,\n    iterations::Int=Int(5e3),show_trace::Bool=true, \n    λ0::Vector{Vector{Float64}}=Vector{Vector{Float64}}())\n\nOptimize parameters specified within fit vectors.\n\n\n\n\n\n"
},

{
    "location": "man/neural_observation_model/#Some-important-functions-1",
    "page": "Fitting a model to neural activity",
    "title": "Some important functions",
    "category": "section",
    "text": "    optimize_model(pz::Vector{TT},py::Vector{Vector{TT}},pz_fit,py_fit,data;\n        dt::Float64=1e-2, n::Int=53, f_str=\"softplus\",map_str::String=\"exp\",\n        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),\n        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),\n        x_tol::Float64=1e-16,f_tol::Float64=1e-16,g_tol::Float64=1e-12,\n        iterations::Int=Int(5e3),show_trace::Bool=true, \n        λ0::Vector{Vector{Float64}}=Vector{Vector{Float64}}()) where {TT <: Any}"
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
    "location": "man/aggregating_sessions/#",
    "page": "Aggregating data from separate recording sessions",
    "title": "Aggregating data from separate recording sessions",
    "category": "page",
    "text": ""
},

{
    "location": "man/aggregating_sessions/#Aggregating-data-from-separate-recording-sessions-1",
    "page": "Aggregating data from separate recording sessions",
    "title": "Aggregating data from separate recording sessions",
    "category": "section",
    "text": "We can fit the parameters of the latent model uses animal choices."
},

{
    "location": "man/aggregating_sessions/#Key-functions-1",
    "page": "Aggregating data from separate recording sessions",
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
    "text": "(Thanks to Nick Roy of the PillowLab for these instructions!)This page is all about how to ssh into spock, della, scotty or any other campus resource without needing to use VPN.The basic idea here is to use RSA to ssh through an intermediate, open, server. In this case, the two options are nobel (nobel.princeton.edu) and arizona (arizona.princeton.edu). This should work on any unix machine (linux and MAC). Windows people should seek immediate attention. The steps are as follows:Edit (or create and edit) an ssh config on your local machine (usually located at ~/.ssh/config), and add the following code:   Host nobel\n   User UNAME\n   HostName nobel.princeton.edu\n   Host spock\n   User UNAME\n   HostName scotty.princeton.edu\n   ForwardX11 yes\n   ForwardX11Trusted yes\n   ProxyCommand ssh nobel -oClearAllForwardings=yes -W %h:%p   In this code, you should replace UNAME with your username (i.e. what you log into each server under) and nobel can be replaced everywhere with arizona if you would like to use arizona as the pass-through server. To access other machines, replace spock with della or scotty.Create RSA keys to facilitate no-password logins more information here. The steps here are to make the RSA key:    >> ssh-keygen -t rsaThen hit enter twice to save the RSA key to the default location and to not include an RSA password. Now add the key to the pass-through server and the remote machine via:    >> ssh-copy-id UNAME@nobel.princeton.edu\n    >> ssh-copy-id UNAME@spock.princeton.eduwhere again UNAME is your login name and you can change the address (spock.princeton.edu) to whichever machine you are trying to access. Make sure to do this first for either arizona or nobel (whichever you decide to use) and then again for the machine you are trying to access.With the above code, you can now simply ssh in via    >> ssh scotty   and not even have to worry about VPN, passwords etc.These instructions can also be found here."
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
    "location": "functions/#pulse_input_DDM.optimize_model-Union{Tuple{TT}, Tuple{Array{TT,1},TT,Any,Any,Any}} where TT",
    "page": "Functions",
    "title": "pulse_input_DDM.optimize_model",
    "category": "method",
    "text": "optimize_model(pz, bias, pz_fit_vec, bias_fit_vec,\n    data; dt, n, map_str, x_tol,f_tol,g_tol, iterations)\n\nOptimize parameters specified within fit vectors.\n\n\n\n\n\n"
},

{
    "location": "functions/#Functions-1",
    "page": "Functions",
    "title": "Functions",
    "category": "section",
    "text": "    optimize_model(pz::Vector{TT}, bias::TT, pz_fit_vec, bias_fit_vec,\n        data; dt::Float64=1e-2, n=53, map_str::String=\"exp\",\n        x_tol::Float64=1e-16,f_tol::Float64=1e-16,g_tol::Float64=1e-12,\n        iterations::Int=Int(5e3)) where {TT <: Any}"
},

]}
