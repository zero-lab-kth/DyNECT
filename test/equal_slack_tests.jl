# Regression tests for the equal-slack certification of Nabetani regions (gne_regions) and for
# Algorithm 1 with and without a leader parameter γ. Every equilibrium used as input is checked
# analytically (comments) and with the independent certificate DyNECT.is_gne.

using DyNECT, Test, LinearAlgebra, CommonSolve, ParametricDAQP, Random, JuMP

# Rosen: J₁ = ½x₁² − x₁x₂, J₂ = x₂² + x₁x₂, x₁ + x₂ ≥ K stored as −x₁ − x₂ ≤ b = −K.
# Best responses: x₁ = max(x₂, K − x₂), x₂ = max(−x₁/2, K − x₁).
# K > 0: GNE set {x₁ + x₂ = K, K/2 ≤ x₁ ≤ 2K}.  K < 0: unique GNE (0, 0), constraint slack.
const QR = [[[1.;;], [-1.;;]], [[1.;;], [2.;;]]]
rosen(b; A_sh=[[-1.;;], [-1.;;]]) = DyNECT.StaticGNEP(QR, [[0.], [0.]], [zeros(0, 1), zeros(0, 1)],
                                                      [Float64[], Float64[]], A_sh, b)
# basic: Jᵢ = ½xᵢ² − 10xᵢ, xᵢ ≥ 0, x₁ + x₂ ≤ 1, 4x₁ + x₂ ≤ 2; GNE set: x₂ = min(1 − x₁, 2 − 4x₁), x₁ ∈ [0, ½]
basic() = DyNECT.StaticGNEP([[[1.;;], [0.;;]], [[0.;;], [1.;;]]], [[-10.], [-10.]], [[-1.;;], [-1.;;]],
                            [[0.], [0.]], [[1.; 4.0;;], [1.0; 1.0;;]], [1.; 2.])
mapof(g; θub, θlb, kw...) = (s = CommonSolve.init(g, DyNECT.NabetaniParametrizationSolver; θub=θub, θlb=θlb, verbose=0, kw...);
                            sol = CommonSolve.solve!(s); (sol=sol, mp=s.mpavi, solver=s))
slacks(g, θ, x) = (offs = cumsum([0; g.n]); xi = [x[offs[i]+1:offs[i+1]] for i in 1:g.N]; m = length(g.b_sh);
                   β = [[θ[(i-1)*m+1:i*m] for i in 1:g.N-1]; [g.b_sh - sum(θ[(i-1)*m+1:i*m] for i in 1:g.N-1)]];
                   [β[i] - g.A_sh[i] * xi[i] for i in 1:g.N])

@testset "Equal-slack certification" begin

    @testset "A. allocation identities (convention, signs, ordering)" begin
        Random.seed!(3)
        N, m, n, nγ = 3, 2, [2, 1, 2], 1
        Q = [[i == j ? Matrix(2.0I, n[i], n[i]) : 0.1 * randn(n[i], n[j]) for j in 1:N] for i in 1:N]
        g = DyNECT.StaticGNEP(Q, [randn(n[i]) for i in 1:N], [zeros(0, n[i]) for i in 1:N], [Float64[] for i in 1:N],
                              [randn(m, n[i]) for i in 1:N], randn(m))
        lp = DyNECT.LeaderParametrizedGNEP(g, randn(sum(n), nγ), randn(m, nγ))
        for _ in 1:20
            x, γ = randn(sum(n)), randn(nγ)
            θ = DyNECT.equal_slack_allocation(lp, x, γ)
            β = [[θ[(i-1)*m+1:i*m] for i in 1:N-1]; [g.b_sh + lp.Bγ * γ - sum(θ[(i-1)*m+1:i*m] for i in 1:N-1)]]
            offs = cumsum([0; n]); xi(i) = x[offs[i]+1:offs[i+1]]
            @test sum(β) ≈ g.b_sh + lp.Bγ * γ                                           # Σβ = b_sh + Bγγ
            s = [β[i] - g.A_sh[i] * xi(i) for i in 1:N]
            @test all(si ≈ (g.b_sh + lp.Bγ * γ - sum(g.A_sh[j] * xi(j) for j in 1:N)) / N for si in s)  # equal slack
            # per-agent form g_i = A_i x_i − (b + Bγγ)/N, Σg ≤ 0: β_i − (b+Bγγ)/N = g_i − mean(g)
            gi = [g.A_sh[i] * xi(i) - (g.b_sh + lp.Bγ * γ) / N for i in 1:N]
            @test all(β[i] - (g.b_sh + lp.Bγ * γ) / N ≈ gi[i] - sum(gi) / N for i in 1:N)
            @test sum(gi[i] - sum(gi) / N for i in 1:N) ≈ zeros(m) atol = 1e-12          # Σ Λ_spec = 0
        end
        # the assembled mpAVI uses p = [θ; γ] and puts b_sh + Bγγ − Σθ on agent N's rows
        mp = DyNECT.NabetaniParametrization(lp; θub=ones(4), θlb=-ones(4), γub=[1.0], γlb=[-1.0])
        p = randn(5); rowsN = (N - 1) * m .+ (1:m)
        @test (mp.B * p + mp.b)[rowsN] ≈ g.b_sh + lp.Bγ * p[5:5] - (p[1:2] + p[3:4])
        @test (mp.B * p + mp.b)[1:m] ≈ p[1:2]
    end

    @testset "B. active, strictly inactive and mixed shared constraints" begin
        # active: Rosen K = 1, x = (0.8, 0.2) is a GNE (0.5 ≤ 0.8 ≤ 2); common slack 0
        g = rosen([-1.0]); M = mapof(g; θub=[5.0], θlb=[-5.0]); R = DyNECT.gne_regions(M.sol, g, M.mp)
        x = [0.8, 0.2]; θ = DyNECT.equal_slack_allocation(g, x)
        @test DyNECT.is_gne(g, x).ok
        @test all(s -> abs(s[1]) < 1e-12, slacks(g, θ, x))
        @test DyNECT.in_gne_map(R, g, x).found
        # strictly inactive: Rosen K = −0.5, unique GNE (0, 0), common slack 0.25 > 0
        g = rosen([0.5]); M = mapof(g; θub=[5.0], θlb=[-5.0]); R = DyNECT.gne_regions(M.sol, g, M.mp)
        x = [0.0, 0.0]; θ = DyNECT.equal_slack_allocation(g, x)
        @test DyNECT.is_gne(g, x).ok
        @test all(s -> s[1] ≈ 0.25, slacks(g, θ, x))
        @test DyNECT.in_gne_map(R, g, x).found
        # mixed: basic example, x = (0.2, 0.8): x₁ + x₂ = 1 (active), 4x₁ + x₂ = 1.6 < 2 (inactive);
        # best responses: x₁ ≤ min(1 − 0.8, (2 − 0.8)/4) = 0.2, x₂ ≤ min(1 − 0.2, 2 − 0.8) = 0.8
        g = basic(); M = mapof(g; θub=[5.0, 5.0], θlb=[-5.0, -5.0]); R = DyNECT.gne_regions(M.sol, g, M.mp)
        x = [0.2, 0.8]; θ = DyNECT.equal_slack_allocation(g, x)
        @test DyNECT.is_gne(g, x).ok
        s = slacks(g, θ, x)
        @test all(si -> abs(si[1]) < 1e-12 && si[2] ≈ 0.2, s)
        @test DyNECT.in_gne_map(R, g, x).found
    end

    @testset "C. degenerate active sets" begin
        # (i) duplicated shared row: the GNE set is unchanged (same feasible sets)
        A2 = [[-1.0; -1.0;;], [-1.0; -1.0;;]]
        g = rosen([-1.0, -1.0]; A_sh=A2)
        M = mapof(g; θub=fill(5.0, 2), θlb=fill(-5.0, 2))
        @test M.sol.status == :Solved
        R = DyNECT.gne_regions(M.sol, g, M.mp)
        ts = range(0.5, 2; length=31)
        @test all(DyNECT.is_gne(g, [t, 1 - t]).ok for t in ts)
        @test all(DyNECT.in_gne_map(R, g, [t, 1 - t]).found for t in ts)            # completeness (numerical)
        @test DyNECT.check_gne_map(M.sol, g, M.mp).ok                                # soundness (numerical)
        # basis independence. In the Nabetani lifting the two copies of the duplicated row are distinct
        # parametric rows (agent 1: −x₁ ≤ θ₁ and −x₁ ≤ θ₂; agent 2: −x₂ ≤ −1 − θ₁ and −x₂ ≤ −1 − θ₂), so a
        # different basis gives a different law off E_r. At a GNE the equal-slack allocation has θ₁ = θ₂ and
        # both copies bind: replacing a basis row by its twin must give the same x and the same certificate.
        n = 2; twin = Dict(1 => 2, 2 => 1, 3 => 4, 4 => 3)                         # rows: agent 1 (1, 2), agent 2 (3, 4)
        law(S) = (Aa = M.mp.A[S, :]; KKT = [M.mp.H Aa'; Aa zeros(length(S), length(S))];
                  Mx = KKT \ [-M.mp.F; M.mp.B[S, :]]; c = KKT \ [-M.mp.f; M.mp.b[S]]; (Mx[1:n, :], c[1:n]))
        nswapped = 0
        for t in ts
            x = [t, 1 - t]
            hit = DyNECT.in_gne_map(R, g, x); p = hit.θ
            @test p[1] ≈ p[2]                                                       # equal allocations of the twins
            AS = only(Rr.AS for Rr in R if Rr.region == hit.region)
            for (k, r) in enumerate(AS)
                twin[r] in AS && continue
                AS2 = copy(AS); AS2[k] = twin[r]
                K2, h2 = law(AS2)
                @test K2 * p + h2 ≈ x atol = 1e-10                                  # same equilibrium
                E2, e2 = DyNECT.equal_slack_constraint(g, K2, h2)
                @test E2 * p ≈ e2 atol = 1e-10                                      # certified under either basis
                nswapped += 1
            end
        end
        @test nswapped > 0
        # (ii) zero row: agent 3 does not enter the shared constraint; GNE set {x₁+x₂ = 1, ½ ≤ x₁ ≤ 2, x₃ = 0}
        g3 = DyNECT.StaticGNEP([[[1.;;], [-1.;;], [0.;;]], [[1.;;], [2.;;], [0.;;]], [[0.;;], [0.;;], [1.;;]]],
                               [[0.], [0.], [0.]], [zeros(0, 1) for _ in 1:3], [Float64[] for _ in 1:3],
                               [[-1.;;], [-1.;;], [0.;;]], [-1.])
        M3 = mapof(g3; θub=[4.4, 4.4], θlb=[-5.6, -5.6], AS0=[1, 2])   # AS0: ParametricDAQP's own start fails here
        R3 = DyNECT.gne_regions(M3.sol, g3, M3.mp)
        @test all(DyNECT.is_gne(g3, [t, 1 - t, 0.0]).ok for t in ts)
        @test all(DyNECT.in_gne_map(R3, g3, [t, 1 - t, 0.0]).found for t in ts)
        @test DyNECT.check_gne_map(M3.sol, g3, M3.mp).ok
        @test DyNECT.filter_gne_crs!(deepcopy(M3.sol), g3) == 0     # legacy (deprecated) filter keeps no region
    end

    @testset "D. unequal positive slacks: Nabetani condition holds, equal slack does not" begin
        # Rosen K = −0.5 (b = 0.5): agent 1: −x₁ ≤ θ, agent 2: −x₂ ≤ 0.5 − θ. For θ ∈ (0, 0.5) the pVI
        # solution is the unconstrained NE (0, 0) with slacks θ and 0.5 − θ, both > 0.
        g = rosen([0.5]); M = mapof(g; θub=[5.0], θlb=[-5.0]); R = DyNECT.gne_regions(M.sol, g, M.mp)
        θ = [0.1]
        x = DyNECT.evaluatePWA(M.sol, θ)
        @test norm(x) < 1e-9
        s = slacks(g, θ, x)
        @test s[1][1] ≈ 0.1 && s[2][1] ≈ 0.4                 # unequal and strictly positive: all copies inactive
        @test DyNECT.is_gne(g, x).ok                         # genuine GNE ...
        @test isnothing(DyNECT.evaluate_gne(R, θ))           # ... but θ is not its equal-slack allocation
        θs = DyNECT.equal_slack_allocation(g, x)
        @test θs ≈ [0.25]
        @test norm(DyNECT.evaluate_gne(R, θs)) < 1e-9        # the same GNE is represented at Λ(x)
    end

    @testset "E. optimal selection against independent solutions" begin
        # Rosen K = 1, SW = ½x₁² + x₂² on x₁ + x₂ = 1 → t = 2/3
        g = rosen([-1.0]); M = mapof(g; θub=[5.0], θlb=[-5.0])
        r = DyNECT.select_optimal_gne(M.sol, g, M.mp, x -> 0.5x[1]^2 + x[2]^2, true; verify=true)
        @test norm(r.u_star - [2 / 3, 1 / 3]) < 1e-8
        @test r.status == :optimal && r.map_status == :Solved && r.convexity == :verified && !r.box_boundary
        @test DyNECT.is_gne(g, r.u_star).ok
        # basic example: projections onto x₂ = min(1 − x₁, 2 − 4x₁), x₁ ∈ [0, ½]
        g = basic(); M = mapof(g; θub=[5.0, 5.0], θlb=[-5.0, -5.0])
        cps(P, A, B) = (d = B - A; A + clamp(dot(P - A, d) / dot(d, d), 0, 1) * d)
        Random.seed!(5)
        for _ in 1:10
            t = rand(2) .* [0.5, 2.0]
            p1, p2 = cps(t, [0.0, 1.0], [1/3, 2/3]), cps(t, [1/3, 2/3], [0.5, 0.0])
            xe = norm(p1 - t) <= norm(p2 - t) ? p1 : p2
            r = DyNECT.select_optimal_gne(M.sol, g, M.mp, x -> sum((x .- t) .^ 2), true)
            @test norm(r.u_star - xe) < 1e-8 && DyNECT.is_gne(g, r.u_star).ok && r.status == :optimal
        end
        # joint leader selection: Rosen with b_sh = −1, Bγ = −1 (K = 1 + γ), Γ = [−0.4, 0.6]
        lp = DyNECT.LeaderParametrizedGNEP(rosen([-1.0]), zeros(2, 1), fill(-1.0, 1, 1))
        s = CommonSolve.init(lp, DyNECT.NabetaniParametrizationSolver; θub=[5.0], θlb=[-5.0], γub=[0.6], γlb=[-0.4], verbose=0)
        sol = CommonSolve.solve!(s)
        @test s.status[] == :Solved
        # independent brute force over the analytic GNE family x = (tK, (1−t)K), t ∈ [½, 2]
        brute(φ) = minimum(φ([t * (1 + γ), (1 - t) * (1 + γ)], [γ]) for γ in range(-0.4, 0.6; length=401), t in range(0.5, 2; length=601))
        # (i) φ = ½x₁² + x₂²: value K²/3 at t = 2/3, minimal at K = 0.6 → γ = −0.4, x = (0.4, 0.2), φ = 0.12
        φ1(x, γ) = 0.5x[1]^2 + x[2]^2
        r = DyNECT.select_optimal_gne(sol, lp, s.mpavi, φ1, true; verify=true)
        @test r.γ_star ≈ [-0.4] atol = 1e-8
        @test norm(r.u_star - [0.4, 0.2]) < 1e-8 && r.φ_star ≈ 0.12
        @test abs(brute(φ1) - 0.12) < 1e-4 && r.φ_star <= brute(φ1) + 1e-12
        @test DyNECT.is_gne(lp, r.u_star, r.γ_star).ok && r.status == :optimal
        # (ii) φ = ½x₁² + x₂² + (γ − ½)²: (1+γ)²/3 + (γ−½)² is minimal at γ = 1/8 → x = (3/4, 3/8), φ = 9/16
        φ2(x, γ) = 0.5x[1]^2 + x[2]^2 + (γ[1] - 0.5)^2
        r = DyNECT.select_optimal_gne(sol, lp, s.mpavi, φ2, true; verify=true)
        @test r.γ_star ≈ [0.125] atol = 1e-8
        @test norm(r.u_star - [0.75, 0.375]) < 1e-8 && r.φ_star ≈ 0.5625
        @test abs(brute(φ2) - 0.5625) < 1e-4 && r.φ_star <= brute(φ2) + 1e-12
        @test DyNECT.is_gne(lp, r.u_star, r.γ_star).ok
        # Γ as an extra polytope γ ≥ 0.2: (ii) is increasing for γ > 1/8, so γ = 0.2, K = 1.2, x = (2K/3, K/3) = (0.8, 0.4)
        r = DyNECT.select_optimal_gne(sol, lp, s.mpavi, φ2, true; Γ=(A=fill(-1.0, 1, 1), b=[-0.2]))
        @test r.γ_star ≈ [0.2] atol = 1e-8
        @test norm(r.u_star - [0.8, 0.4]) < 1e-8
    end

    @testset "F. edge cases and statuses" begin
        # point and empty certified sets (Rosen K = −0.5): one point region, the others infeasible
        g = rosen([0.5]); M = mapof(g; θub=[5.0], θlb=[-5.0])
        chk = DyNECT.check_gne_map(M.sol, g, M.mp)
        @test chk.ok && all(r.dim == 0 for r in chk.regions)
        r = DyNECT.select_optimal_gne(M.sol, g, M.mp, x -> sum(abs2, x), true)
        @test norm(r.u_star) < 1e-9 && r.status == :optimal
        @test count(==(:solved), values(r.region_status)) >= 1
        @test any(st -> st in (:infeasible, :empty_inconsistent), values(r.region_status))
        @test all(st -> st in (:solved, :infeasible, :empty_inconsistent), values(r.region_status))
        # numerical coverage check of the enumeration (evidence, not a proof): every grid point of the box,
        # boundary included, with Y(θ) ≠ ∅ lies in some closed critical region
        @test all(!isnothing(DyNECT.evaluatePWA(M.sol, [t])) for t in range(-5, 5; length=1001))
        # nonquadratic objective: Ipopt path; convexity neither verified nor assumed → :unverified
        g = rosen([-1.0]); M = mapof(g; θub=[5.0], θlb=[-5.0])
        r = DyNECT.select_optimal_gne(M.sol, g, M.mp, x -> sqrt(1 + (x[1] - 1)^2), false)
        @test r.status == :unverified && r.convexity == :unknown && norm(r.u_star - [1.0, 0.0]) < 1e-6
        r = DyNECT.select_optimal_gne(M.sol, g, M.mp, x -> sqrt(1 + (x[1] - 1)^2), false; assume_convex=true)
        @test r.status == :optimal && r.convexity == :assumed
        # known nonconvex quadratic: −(x₁ − 1)² on the GNE segment x₁ ∈ [½, 2] (reduced Hessian < 0) is not
        # sent to Clarabel; the regions are solved locally with Ipopt and the status is :unverified. The
        # endpoints are separate point regions, so the best candidate is the far endpoint x = (2, −1).
        r = DyNECT.select_optimal_gne(M.sol, g, M.mp, x -> -(x[1] - 1)^2, true)
        @test r.convexity == :nonconvex && r.status == :unverified
        @test norm(r.u_star - [2.0, -1.0]) < 1e-6 && DyNECT.is_gne(g, r.u_star).ok
        r = DyNECT.select_optimal_gne(M.sol, g, M.mp, x -> -(x[1] - 1)^2, true; assume_convex=true)
        @test r.convexity == :nonconvex && r.status == :unverified          # a detected nonconvexity overrides the assumption
        # incomplete enumeration propagates to the selection status
        lp = DyNECT.LeaderParametrizedGNEP(rosen([-1.0]), zeros(2, 1), fill(-1.0, 1, 1))
        s = CommonSolve.init(lp, DyNECT.NabetaniParametrizationSolver; θub=[5.0], θlb=[-5.0], γub=[0.6], γlb=[-0.4],
                             verbose=0, region_limit=1)
        sol = CommonSolve.solve!(s)
        @test s.status[] != :Solved
        r = DyNECT.select_optimal_gne(sol, lp, s.mpavi, (x, γ) -> 0.5x[1]^2 + x[2]^2, true)
        @test r.status == :partial && r.map_status == s.status[]
    end

    @testset "G. parameter-box limitation (represented vs. original GNE set)" begin
        # Rosen K = 1: GNE set x₁ ∈ [½, 2], equal-slack allocation θ = Λ(x) = −x₁ ∈ [−2, −½].
        # With θ ∈ [−1.5, 5], GNEs with x₁ > 1.5 are genuine but not represented.
        g = rosen([-1.0]); M = mapof(g; θub=[5.0], θlb=[-1.5]); R = DyNECT.gne_regions(M.sol, g, M.mp)
        x = [1.8, -0.8]
        @test DyNECT.is_gne(g, x).ok
        @test DyNECT.equal_slack_allocation(g, x) ≈ [-1.8]
        @test !DyNECT.in_gne_map(R, g, x).found                       # excluded by the box
        @test DyNECT.in_gne_map(R, g, [1.5, -0.5]).found               # boundary GNE still represented
        # selection for target (2, −1): optimal over the represented set only, flagged on the box boundary
        φ(x) = (x[1] - 2)^2 + (x[2] + 1)^2
        r = DyNECT.select_optimal_gne(M.sol, g, M.mp, φ, true)
        @test norm(r.u_star - [1.5, -0.5]) < 1e-8 && r.status == :optimal && r.box_boundary
        # with a box containing every GNE allocation the true optimum (2, −1) is found and not flagged
        M = mapof(g; θub=[5.0], θlb=[-5.0])
        r = DyNECT.select_optimal_gne(M.sol, g, M.mp, φ, true)
        @test norm(r.u_star - [2.0, -1.0]) < 1e-8 && r.status == :optimal && !r.box_boundary
    end

    @testset "H. status semantics and safeguards" begin
        g = rosen([-1.0])
        # PWAConvexOptSolver reports :Solved only for a selection optimal over the represented set
        o = CommonSolve.init(OptimalGNEP(g, x -> 0.5x[1]^2 + x[2]^2), DyNECT.PWAConvexOptSolver; θub=[5.0], θlb=[-5.0], verbose=0)
        res = CommonSolve.solve!(o)
        @test o.status[] == :Solved && res.result.status == :optimal
        o = CommonSolve.init(OptimalGNEP(g, x -> -(x[1] - 1)^2), DyNECT.PWAConvexOptSolver; θub=[5.0], θlb=[-5.0], verbose=0)
        res = CommonSolve.solve!(o)
        @test o.status[] == :unverified == res.result.status
        # a failed numerical certificate downgrades :optimal: certify the selected point of K = 1 on the game with
        # K = 1.5, where x = (2/3, 1/3) is infeasible
        M = mapof(g; θub=[5.0], θlb=[-5.0])
        r = DyNECT.select_optimal_gne(M.sol, g, M.mp, x -> 0.5x[1]^2 + x[2]^2, true; verify=true, game_at=γ -> rosen([-1.5]))
        @test r.status == :unverified && norm(r.u_star - [2 / 3, 1 / 3]) < 1e-8
        # no equal-slack allocation of a GNE in the box (θ = −x₁ ∈ [−2, −½], box [1, 5]): distinct error
        M = mapof(g; θub=[5.0], θlb=[1.0])
        err = try DyNECT.select_optimal_gne(M.sol, g, M.mp, x -> sum(abs2, x), true); "" catch e; sprint(showerror, e) end
        @test occursin("no GNE is represented", err)
        # regional Hessian indefinite in (θ, γ) but positive definite on the null space of the equality:
        # leader Rosen (K = 1 + γ, γ ∈ [−0.4, 0.6]), φ = 2x₁² − 0.2γ². The two outer regions have a segment E_r:
        #   x₁ = x₂ = K/2:      φ = K²/2 − 0.2γ²,  φ'' = 0.6 > 0, φ' = 1 + 0.6γ > 0  → γ = −0.4, x = (0.3, 0.3),  φ = 0.148
        #   x₁ = 2K, x₂ = −K:   φ = 8K² − 0.2γ²,   φ'' = 15.6 > 0, φ' = 16K − 0.4γ > 0 → γ = −0.4, x = (1.2, −0.6), φ = 2.848
        lp = DyNECT.LeaderParametrizedGNEP(g, zeros(2, 1), fill(-1.0, 1, 1))
        s = CommonSolve.init(lp, DyNECT.NabetaniParametrizationSolver; θub=[5.0], θlb=[-5.0], γub=[0.6], γlb=[-0.4], verbose=0)
        sol = CommonSolve.solve!(s)
        φ(x, γ) = 2x[1]^2 - 0.2γ[1]^2
        r = DyNECT.select_optimal_gne(sol, lp, s.mpavi, φ, true)
        nseg = 0
        for R in DyNECT.gne_regions(sol, lp, s.mpavi)
            size(R.E, 1) == 1 || continue
            mdl = JuMP.Model(); JuMP.@variable(mdl, pp[1:2])
            P, _ = DyNECT.quadratic_coefficients(φ(R.K * pp .+ R.h, pp[2:2]), pp)
            Z = nullspace(R.E)
            (minimum(eigvals(Symmetric(P))) < 0 && minimum(eigvals(Symmetric(Z' * P * Z))) > 0) || continue
            nseg += 1
            c = only(c for c in r.all_candidates if c.region == R.region)
            xe, φe = abs(c.u[1] - c.u[2]) < 1e-6 ? ([0.3, 0.3], 0.148) : ([1.2, -0.6], 2.848)
            @test c.convexity == :verified && c.status == :solved
            @test norm(c.u - xe) < 1e-8
            @test c.γ ≈ [-0.4] atol = 1e-8
            @test c.φ ≈ φe
        end
        @test nseg == 2
        @test r.convexity == :nonconvex && r.status == :unverified   # the middle region is genuinely nonconvex
        @test all(c.eq_residual <= 1e-9 for c in r.all_candidates)        # unreduced equal-slack equality
        # a solver reporting OPTIMAL at θ = 0, outside every E_r: no infeasible candidate may be selected
        MOIU = JuMP.MOI.Utilities
        lying() = (m = MOIU.MockOptimizer(MOIU.UniversalFallback(MOIU.Model{Float64}()));
                   MOIU.set_mock_optimize!(m, mm -> MOIU.mock_optimize!(mm, JuMP.MOI.OPTIMAL, (JuMP.MOI.FEASIBLE_POINT, [0.0]))); m)
        M = mapof(g; θub=[5.0], θlb=[-5.0])
        r = DyNECT.select_optimal_gne(M.sol, g, M.mp, x -> sum(abs2, x), true, lying)
        R = Dict(R.region => R for R in DyNECT.gne_regions(M.sol, g, M.mp))
        @test all(DyNECT.regional_residual(c.θ, R[c.region], R[c.region].Ain, R[c.region].bin) <= 1e-6 for c in r.all_candidates)
        @test DyNECT.is_gne(g, r.u_star).ok && r.status == :partial
        @test :infeasible_candidate in values(r.region_status) && :inexact in values(r.region_status)
    end

    @testset "I. stored ParametricDAQP v0.6.1 law vs independent VI solution" begin
        # region with a multiplier constant in θ: normalize_model sets its factor to 0, so the stored law drops
        # that constraint; the KKT law of gne_regions must solve the VI, x = proj_Y(x - (Hx + Fθ + f))
        Q = [[[2.0 0.5; 0.5 1.5], reshape([0.4, -0.3], 2, 1)], [[0.2 0.6], fill(2.0, 1, 1)]]
        g0 = StaticGNEP(Q, [[-1.0, -2.0], [-1.5]], [-Matrix(1.0I, 2, 2), fill(1.0, 1, 1)], [[0.0, 0.0], [2.0]], [[1.0 1.0], fill(1.0, 1, 1)], [1.0])
        g = DyNECT.fix_leader(DyNECT.LeaderParametrizedGNEP(g0, reshape([1.0, 0.5, -0.8], 3, 1), fill(2.0, 1, 1)), [0.75])
        M = mapof(g; θub=[3.0], θlb=[-2.0])
        function vi_res(θ, x)
            md = JuMP.Model(DyNECT.Clarabel.Optimizer); JuMP.set_silent(md); JuMP.@variable(md, y[1:3])
            JuMP.@constraint(md, M.mp.A * y .<= M.mp.B * θ + M.mp.b)
            JuMP.@objective(md, Min, sum((y - x + M.mp.H * x + M.mp.F * θ + M.mp.f) .^ 2)); JuMP.optimize!(md)
            norm(JuMP.value.(y) - x, Inf)
        end
        res_kkt, res_stored = 0.0, 0.0
        for R in DyNECT.gne_regions(M.sol, g, M.mp; include_inconsistent=true), θ in ([t] for t in range(-2, 3; length=101))
            all(R.G * θ .<= R.g .- 1e-6) || continue
            res_kkt = max(res_kkt, vi_res(θ, R.K * θ + R.h))
            res_stored = max(res_stored, vi_res(θ, DyNECT.evaluatePWA(M.sol, θ)))
        end
        @test res_kkt < 1e-6
        @test_broken res_stored < 1e-6   # unexpected pass: the dependency is fixed and cr.z can be used again
    end
end
