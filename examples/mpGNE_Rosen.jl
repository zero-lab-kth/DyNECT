# Implements the example GNE problem from Rosen, 1965

using DyNECT
using CommonSolve
using LinearAlgebra
using ParametricDAQP
using JuMP, Ipopt
using Plots

# J₁ = .5(x₁)² - x₁x₂
# J₂ = (x₂)² + x₁x₂
# x₁ + x₂ ≥ K := 1 + γ

Q = [[[1.;;], [-1.;;]],
    [[1.;;], [2.;;]]]
q = [[0.],
    [0.]]
A_loc = [zeros(0, 1), zeros(0, 1)]
b_loc = [Float64[], Float64[]]
A_sh = [[-1.;;], [-1.;;]]
rosen(γ) = DyNECT.StaticGNEP(Q, q, A_loc, b_loc, A_sh, [-(1.0 + γ)])

J(x) = [0.5x[1]^2 - x[1]x[2], x[2]^2 + x[1]x[2]]

θub = [5.0]
θlb = [-5.0]
tol = 1e-6
all_test_ok = true

for γ in (-0.4, 0.0, 0.6)
    global all_test_ok
    K = 1.0 + γ
    gnep = rosen(γ)

    solver = CommonSolve.init(gnep, DyNECT.NabetaniParametrizationSolver; θub=θub, θlb=θlb, verbose=0)
    sol = CommonSolve.solve!(solver)
    regions = DyNECT.gne_regions(sol, gnep, solver.mpavi)

    x_pvi, x_gne = Vector{Float64}[], Vector{Float64}[]
    for θ in range(θlb[1], θub[1]; length=2001)
        x = DyNECT.evaluatePWA(sol, [θ])
        isnothing(x) || push!(x_pvi, x)
        x = DyNECT.evaluate_gne(regions, [θ])
        isnothing(x) || push!(x_gne, x)
    end
    on_gne_set(x) = abs(x[1] + x[2] - K) < tol && K / 2 - tol <= x[1] <= 2K + tol
    thm1_ok = all(on_gne_set, x_gne) &&
              minimum(x[1] for x in x_gne) < K / 2 + 1e-2 &&
              maximum(x[1] for x in x_gne) > 2K - 1e-2
    n_spurious = count(!on_gne_set, x_pvi)

    res_sw = CommonSolve.solve(DyNECT.OptimalGNEP(gnep, x -> sum(J(x))), DyNECT.PWAConvexOptSolver;
                               θub=θub, θlb=θlb, verbose=0)
    t_sw = res_sw.x[1] / K
    sw_ok = norm(res_sw.x - [2K / 3, K / 3]) < tol

    best = (φ=Inf, x=Float64[])
    for R in regions
        md = Model(Ipopt.Optimizer); set_silent(md); set_attribute(md, "tol", 1e-12)
        @variable(md, θlb[1] <= θ[1:1] <= θub[1]); @variable(md, s)
        @constraint(md, R.G * θ .<= R.g); @constraint(md, R.E * θ .== R.e)
        x = R.K * θ .+ R.h
        @constraint(md, 0.5x[1]^2 - x[1] * x[2] <= s); @constraint(md, x[2]^2 + x[1] * x[2] <= s)
        @objective(md, Min, s); optimize!(md)
        is_solved_and_feasible(md) || continue
        xs = R.K * value.(θ) + R.h
        maximum(J(xs)) < best.φ && (best = (φ=maximum(J(xs)), x=xs))
    end
    t_mm = best.x[1] / K
    mm_ok = abs(t_mm - sqrt(2 / 3)) < 1e-5 && abs(sum(best.x) - K) < tol

    n_full = count(R -> count(θ -> !isnothing(DyNECT.evaluate_gne([R], [θ])), range(θlb[1], θub[1]; length=2001)) > 1, regions)
    println("γ = $γ: $(length(sol.CRs)) regions, $n_full with a full-dimensional E_r; ",
            "Thm 1 $(thm1_ok ? "ok" : "FAIL") ($n_spurious pVI points off the GNE set excluded); ",
            "SW t = $(round(t_sw; digits=6)) (2/3) $(sw_ok ? "ok" : "FAIL"); ",
            "MM t = $(round(t_mm; digits=6)) (√(2/3) = $(round(sqrt(2/3); digits=6))) $(mm_ok ? "ok" : "FAIL")")
    all_test_ok &= thm1_ok && sw_ok && mm_ok

    if γ == 0.0
        plt = scatter(first.(x_pvi), last.(x_pvi); xlabel="x₁", ylabel="x₂", label="pVI solutions", ms=2, color=:lightgray)
        scatter!(plt, first.(x_gne), last.(x_gne); label="GNEs (θ ∈ ∪E_r)", ms=2, color=:blue)
        scatter!(plt, [res_sw.x[1]], [res_sw.x[2]]; label="SW", ms=6, color=:green)
        scatter!(plt, [best.x[1]], [best.x[2]]; label="MM", ms=6, color=:orange)
        scatter!(plt, [K], [0.0]; label="v-GNE", ms=6, color=:red)
        display(plt)
    end
end

if all_test_ok
    println("The parametric solution and Algorithm 1 are correct.")
else
    println("The parametric solution or Algorithm 1 is NOT correct.")
end
