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
    "text": "Code for fitting latent drift diffusion models to pulsed input data and neural activity or behavioral observation data.Pages = [\n    \"man/using_spock.md\",\n    \"man/aggregating_sessions.md\",\n    \"man/choice_observation_model.md\",\n    \"man/neural_observation_model.md\"]\nDepth = 2"
},

{
    "location": "#pulse_input_DDM.optimize_model-Union{Tuple{TT}, Tuple{Array{TT,1},TT,Any,Any,Any}} where TT",
    "page": "Home",
    "title": "pulse_input_DDM.optimize_model",
    "category": "method",
    "text": "optimize_model(pz, bias, pz_fit_vec, bias_fit_vec,\n    data; dt, n, map_str, x_tol,f_tol,g_tol, iterations)\n\nOptimize parameters specified within fit vectors.\n\n\n\n\n\n"
},

{
    "location": "#Functions-1",
    "page": "Home",
    "title": "Functions",
    "category": "section",
    "text": "    optimize_model(pz::Vector{TT}, bias::TT, pz_fit_vec, bias_fit_vec,\n        data; dt::Float64=1e-2, n=53, map_str::String=\"exp\",\n        x_tol::Float64=1e-16,f_tol::Float64=1e-16,g_tol::Float64=1e-12,\n        iterations::Int=Int(5e3)) where {TT <: Any}"
},

{
    "location": "#Index-1",
    "page": "Home",
    "title": "Index",
    "category": "section",
    "text": ""
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

]}
