julia --project=. -e 'cd("examples"); using  NBInclude; @nbinclude("fit_choice_synthetic_data.ipynb"); exit();'
julia --project=. -e 'cd("examples"); using  NBInclude; @nbinclude("fit_joint_synthetic_data.ipynb"); exit();'
julia --project=. -e 'cd("examples"); using  NBInclude; @nbinclude("fit_neural_real_data.ipynb"); exit();'
julia --project=. -e 'cd("examples"); using  NBInclude; @nbinclude("loading_and_saving.ipynb"); exit();'