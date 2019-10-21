using Test, pulse_input_DDM, LinearAlgebra

pz, pd = default_parameters(generative=true)
ntrials, use_bin_center, rng = 10, false, 1
data = sample_inputs_and_choices(pz["generative"], pd["generative"], ntrials; rng=rng)
data = bin_clicks!(data,use_bin_center)

@test compute_LL(pz, pd, data; state="generative") ≈ -3.5184243553389605
@test norm(compute_gradient(pz, pd, data; state="generative")) ≈ 5.614918559409721