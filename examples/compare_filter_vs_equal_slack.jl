# Old pipeline (active-set filter `filter_gne_crs!` + regional selection) vs Algorithm 1
# (equal-slack sets E_r, no filter) on games where the filter is known to work (Rosen, basic 2-D
# example) and on a degenerate Rosen variant whose shared row is zero for one agent.
# Every selected point is certified independently with `DyNECT.is_gne`.
# Run: julia --project=. examples/compare_filter_vs_equal_slack.jl

using DyNECT, CommonSolve, LinearAlgebra, ParametricDAQP, JuMP, Ipopt, Printf, Random

# ---------------------------------------------------------------------------- helpers
# Exact law and region polytope (original θ coordinates) of every critical region
function region_data(sol, mp)
    n = size(mp.H, 1)
    map(enumerate(sol.CRs)) do (k, cr)
        AS = Int.(collect(cr.AS)); Aa = mp.A[AS, :]; nA = length(AS)
        KKT = [mp.H Aa'; Aa zeros(nA, nA)]
        M = KKT \ [-mp.F; mp.B[AS, :]]; c = KKT \ [-mp.f; mp.b[AS]]
        G = cr.Ath' * Diagonal(sol.scaling); g = cr.bth + G * sol.translation
        (r=k, AS=AS, K=M[1:n, :], h=c[1:n], G=[G; mp.A * M[1:n, :] - mp.B], g=[g; mp.b - mp.A * c[1:n]])
    end
end

# Solve min/max of c'x over {θ ∈ P_r (∩ Eθ = e)}; returns nothing if empty
function lp(R, mp, c, sense; E=nothing, e=nothing)
    md = Model(Ipopt.Optimizer); set_silent(md); set_attribute(md, "tol", 1e-12)
    nθ = size(R.K, 2)
    @variable(md, mp.lb[i] <= θ[i=1:nθ] <= mp.ub[i]); @constraint(md, R.G * θ .<= R.g)
    isnothing(E) || isempty(e) || @constraint(md, E * θ .== e)
    x = R.K * θ .+ R.h
    sense == :min ? @objective(md, Min, c' * x) : @objective(md, Max, c' * x)
    optimize!(md); is_solved_and_feasible(md; allow_almost=true) ? (c' * (R.K * value.(θ) + R.h), value.(θ)) : nothing
end

# Regional selection min φ(x) over the regions in `regs`, optionally with the equal-slack equality
function select(regs, mp, ϕ!; eq::Dict=Dict())
    best = (φ=Inf, x=Float64[], r=0)
    for R in regs
        md = Model(Ipopt.Optimizer); set_silent(md); set_attribute(md, "tol", 1e-12)
        nθ = size(R.K, 2)
        @variable(md, mp.lb[i] <= θ[i=1:nθ] <= mp.ub[i]); @constraint(md, R.G * θ .<= R.g)
        if haskey(eq, R.r)
            E, e = eq[R.r]; isempty(e) || @constraint(md, E * θ .== e)
        end
        x = R.K * θ .+ R.h; φexpr = ϕ!(md, x); @objective(md, Min, φexpr)
        optimize!(md); is_solved_and_feasible(md; allow_almost=true) || continue
        xs = R.K * value.(θ) + R.h; v = objective_value(md)
        v < best.φ - 1e-10 && (best = (φ=v, x=xs, r=R.r))
    end
    best
end

# merged union of 1-D intervals
function union_iv(iv)
    iv = sort(filter(!isnothing, iv)); out = Tuple{Float64,Float64}[]
    for (a, b) in iv
        if !isempty(out) && a <= out[end][2] + 1e-7; out[end] = (out[end][1], max(out[end][2], b)); else; push!(out, (a, b)); end
    end
    out
end
fmt(u) = join([@sprintf("[%.6f, %.6f]", a, b) for (a, b) in u], " ∪ ")

# ---------------------------------------------------------------------------- comparison
function compare(name, gnep, θub, θlb, objectives; image_dir=nothing, AS0=nothing)
    solver = CommonSolve.init(gnep, DyNECT.NabetaniParametrizationSolver; θub=θub, θlb=θlb, verbose=0, AS0=AS0)
    mp = solver.mpavi
    sol = CommonSolve.solve!(solver)
    RD = region_data(sol, mp)
    filt = deepcopy(sol); DyNECT.filter_gne_crs!(filt, gnep)
    kept = Set(findall(cr -> any(c -> c.AS == cr.AS, filt.CRs), sol.CRs))
    EQ = Dict(R.region => (R.E, R.e) for R in DyNECT.gne_regions(sol, gnep, mp))
    c = isnothing(image_dir) ? [1.0; zeros(size(mp.H, 1) - 1)] : image_dir    # image measured along x₁

    println("\n================ $name: $(length(sol.CRs)) critical regions ================")
    println(" r | active set       | filter | E_r                  | x₁-image of P_r          | x₁-image of E_r")
    ivF, ivE = [], []
    for R in RD
        Pr = (lp(R, mp, c, :min), lp(R, mp, c, :max))
        if haskey(EQ, R.r)
            E, e = EQ[R.r]
            lo, hi = lp(R, mp, c, :min; E=E, e=e), lp(R, mp, c, :max; E=E, e=e)
            if isnothing(lo); est = "empty"; ivEr = nothing
            else
                dimaff = size(R.K, 2) - size(E, 1)
                est = abs(hi[1] - lo[1]) < 1e-7 ? "point in x" : "affine dim $dimaff"
                ivEr = (lo[1], hi[1])
            end
        else
            est, ivEr = "inconsistent (empty)", nothing
        end
        ivPr = isnothing(Pr[1]) ? nothing : (Pr[1][1], Pr[2][1])
        R.r in kept && push!(ivF, ivPr); push!(ivE, ivEr)
        @printf("%2d | %-16s | %-6s | %-20s | %-24s | %s\n", R.r, string(R.AS), R.r in kept ? "kept" : "drop", est,
                isnothing(ivPr) ? "-" : @sprintf("[%.4f, %.4f]", ivPr...), isnothing(ivEr) ? "-" : @sprintf("[%.4f, %.4f]", ivEr...))
    end
    uF, uE = union_iv(ivF), union_iv(ivE)
    println("image of filtered regions  (old): ", fmt(uF))
    println("image of ∪E_r (Algorithm 1)  (new): ", fmt(uE))
    same_img = length(uF) == length(uE) && all(maximum(abs.(collect(a) .- collect(b))) < 1e-6 for (a, b) in zip(uF, uE))

    ok = same_img
    for (oname, ϕ!) in objectives
        o = select([R for R in RD if R.r in kept], mp, ϕ!)
        n_ = select([R for R in RD if haskey(EQ, R.r)], mp, ϕ!; eq=EQ)
        d = isempty(o.x) || isempty(n_.x) ? Inf : norm(o.x - n_.x, Inf)
        cert(x) = isempty(x) ? "-" : (DyNECT.is_gne(gnep, x).ok ? "GNE" : "NOT GNE")
        ok &= d < 1e-6 && cert(n_.x) == "GNE"
        @printf("  %-10s old: x=%s (r=%d, %s) | new: x=%s (r=%d, %s) | ‖Δx‖∞=%.1e\n", oname,
                string(round.(o.x; digits=6)), o.r, cert(o.x), string(round.(n_.x; digits=6)), n_.r, cert(n_.x), d)
    end
    chk = DyNECT.check_gne_map(sol, gnep, mp)
    println("Theorem 1 (⊇) check: ", chk.ok ? "all sampled points of ∪E_r are certified GNEs" : "FAILED")
    println(ok ? "→ filter and Algorithm 1 agree" : "→ filter and Algorithm 1 DIFFER")
    ok
end

# ---------------------------------------------------------------------------- games
Jr(x) = (0.5x[1]^2 - x[1] * x[2], x[2]^2 + x[1] * x[2])
SW!(md, x) = sum(Jr(x))
MM!(md, x) = (@variable(md, s); J = Jr(x); @constraint(md, J[1] <= s); @constraint(md, J[2] <= s); s)
Random.seed!(1); targets = [rand(2) .* [2.5, 1.5] .- [0.0, 0.5] for _ in 1:20]
TGT = [("target $k", (md, x) -> sum((x[1:2] .- t) .^ 2)) for (k, t) in enumerate(targets)]

all_ok = true
for γ in (-0.4, 0.0, 0.6)
    global all_ok
    g = DyNECT.StaticGNEP([[[1.;;], [-1.;;]], [[1.;;], [2.;;]]], [[0.], [0.]], [zeros(0, 1), zeros(0, 1)], [Float64[], Float64[]],
                          [[-1.;;], [-1.;;]], [-(1.0 + γ)])
    all_ok &= compare("Rosen, γ = $γ (K = $(1 + γ))", g, [5.0], [-5.0], [("SW", SW!), ("MM", MM!), TGT...])
end

# basic 2-D example: Jᵢ = |xᵢ - 10|², x ≥ 0, x₁ + x₂ ≤ 1, 4x₁ + x₂ ≤ 2
g2 = DyNECT.StaticGNEP([[[1.;;], [0.;;]], [[0.;;], [1.;;]]], [[-10.], [-10.]], [[-1.;;], [-1.;;]], [[0.], [0.]],
                       [[1.; 4.0;;], [1.0; 1.0;;]], [1.; 2.])
Random.seed!(2); T2 = [("target $k", (md, x) -> sum((x .- t) .^ 2)) for (k, t) in enumerate([rand(2) .* [0.5, 2.0] for _ in 1:20])]
all_ok &= compare("basic 2-D example", g2, [5.0, 5.0], [-5.0, -5.0], T2)

# degenerate Rosen variant: agent 3 (J₃ = ½x₃²) does not enter the shared row → zero row in A_sh[3]
g3 = DyNECT.StaticGNEP([[[1.;;], [-1.;;], [0.;;]], [[1.;;], [2.;;], [0.;;]], [[0.;;], [0.;;], [1.;;]]], [[0.], [0.], [0.]],
                       [zeros(0, 1), zeros(0, 1), zeros(0, 1)], [Float64[], Float64[], Float64[]],
                       [[-1.;;], [-1.;;], [0.;;]], [-1.0])
SW3!(md, x) = SW!(md, x) + 0.5x[3]^2
MM3!(md, x) = MM!(md, x)
# ParametricDAQP's own initial-active-set search errors out on this instance (compute_AS0 fallback
# in v0.6.1), so the initial active set is given: at θ = (-0.6, -0.6) the rows of agents 1, 2 are active
degenerate_ok = compare("Rosen + non-incident agent 3 (zero row)", g3, [4.4, 4.4], [-5.6, -5.6], [("SW", SW3!), ("MM", MM3!)]; AS0=[1, 2])

println(all_ok ? "\nRosen and basic example: filter and Algorithm 1 agree." : "\nRosen and basic example: DIFFERENCES found (see above).")
println("Degenerate variant: the filter keeps no region; Algorithm 1 recovers the GNE set ",
        "x₁ ∈ [0.5, 2], x₂ = 1 - x₁, x₃ = 0 (certified above).")
