"""
"""
neural_null(k,λ,dt) = sum(logpdf.(Poisson.(λ*dt),k))