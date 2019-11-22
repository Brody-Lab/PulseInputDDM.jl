struct DDMProblem
    data::Dict{Any,Any}
end

function (problem::DDMProblem)(θ)
    pz, pd = θ
    @unpack data = problem
    compute_LL(collect(pz), collect(pd), data)
end

function problem_transformation(problem::DDMProblem)

    tz = as((as(Real, 0., 2.), as(Real, 8., 30.),
            as(Real, -5, 5), as(Real, 0., 100.), as(Real, 0., 2.5),
            as(Real, 0.01, 1.2), as(Real, 0.005, 1.)))

    td = as((as(Real, -30., 30.), as(Real, 0., 1.)))

    as((tz,td))
end
