####### Type conversion functions #####

@doc raw"""
    DynLQGame2mpAVI(prob::DynLQGame, T_hor::Int64)
    Constructs the parametric variational inequality (mpVI) for a dynamic Nash equilibrium problem (DynLQGame) over a finite prediction horizon.

# Arguments
- `prob::DynLQGame`: Dynamic game structure containing system dynamics, cost, and constraints.
- `T_hor::Int64`: Prediction horizon.

# Returns
- `MPVI`: An instance of `ParametricDAQP.MPVI` representing the parametric variational inequality:
    - `H`: Block matrix for the quadratic part of the VI mapping.
    - `F`: Matrix mapping the initial state to the affine part of the VI mapping.
    - `f`: Constant affine vector in the VI mapping.
    - `D`: Constraint matrix for the stacked input sequence.
    - `E`: Matrix mapping the initial state to the affine part of the constraints.
    - `d`: Constant affine vector in the constraints.

The VI is of the form:  
``H u + F x_0 + f``,  subject to  ``D u \leq E x_0 + d``
where ``u`` is the stacked input sequence for all agents.
"""
function DynLQGame2mpAVI(prob::DynLQGame, T_hor::Int64)
    # Matrices defined as in Baghdalhorani, Benenati, Grammatico - Arxiv 2025

    # prediction model: x̅ = Θx₀+(∑ Γᵢu̅ᵢ) + c̅
    Γ, Γi, Θ, c̅ = generate_prediction_model(prob.A, prob.B, T_hor; c=prob.c)

    # Define Q̅[i] = blkdg(I ⊗ Q[i], Pi)
    Q̅ = [BlockDiagonal([kron(I(T_hor - 1), prob.Q[i]), prob.P[i]]) for i in 1:prob.N]

    # Define R̅ as a block matrix where each block is R̅ᵢⱼ = I ⊗ Rᵢⱼ
    R̅ = BlockArray{Float64}(undef_blocks, [T_hor * prob.nu[i] for i = 1:prob.N], [T_hor * prob.nu[i] for i = 1:prob.N])
    for i in 1:prob.N
        for j in 1:prob.N
            R̅[Block(i, j)] = kron(I(T_hor), prob.R[i][j])
        end
    end

    # Define H (linear part of the VI mapping)
    H = BlockArray{Float64}(undef_blocks, [T_hor * prob.nu[i] for i = 1:prob.N], [T_hor * prob.nu[i] for i = 1:prob.N])
    for i in 1:prob.N
        for j in 1:prob.N
            H[Block(i, j)] = Γi[i]' * Q̅[i] * Γi[j]
        end
    end
    H = H + R̅

    # Define F (maps from x0 to the affine part of the VI mapping)
    F = vcat([Γi[i]' * Q̅[i] * Θ for i in 1:prob.N]...)

    # define f (affine part of the VI mapping)
    q̅ = [vcat(kron(ones(T_hor - 1), prob.q[i]), prob.p[i]) for i in 1:prob.N]
    r̅ = [kron(ones(T_hor), prob.r[i]) for i in 1:prob.N]
    f = vcat([Γi[i]' * (Q̅[i] * c̅ + q̅[i]) + r̅[i] for i in 1:prob.N]...)

    ## Constraints
    # Cₓ*x[t] ≤ bₓ ∀ t ==> C̅ₓΓu̅ <= b̅ₓ -C̅ₓΘx₀ - C̅ₓc̅
    # where C̅ₓ = I ⊗ Cₓ
    C̅_x = kron(I(T_hor), prob.C_x)
    # D_shar = row_stack([I ⊗ Du_i ; (I ⊗ Dx)*Γi ] )
    D_shar = hcat([[kron(I(T_hor), prob.C_u_i[i]);
        C̅_x * Γi[i]] for i in 1:prob.N]...)
    # Append local constraints 
    D = [D_shar;
        BlockDiagonal([kron(I(T_hor), prob.C_loc_i[i]) for i = 1:prob.N])]

    # Define E (maps from x0 to constraints)
    E = [zeros(T_hor * prob.m_u, prob.nx)        # Shared input constraints
        -1 * C̅_x * Θ;                            # State constraints
        zeros(sum(prob.m_loc) * T_hor, prob.nx)] # Local input constraints

    # Affine part of the constraints
    d = [kron(ones(T_hor), prob.b_u);            # Shared input constraints
        kron(ones(T_hor), prob.b_x) - C̅_x * c̅    # State constraints
        vcat([kron(ones(T_hor), prob.b_loc_i[i]) for i in 1:prob.N]...)] # Local input constraints
    # VI(H*x + F*x0 + f, D*x <= E*x0 + d)
    return mpAVI(Matrix{Float64}(H), F, f, D, E, d)
end

function DynLQGame2mpAVI(prob::DynLQGameTV) # time varying
    # prediction model: x̅ = Θx₀+(∑ Γᵢu̅ᵢ) + c̅
    Γ, Γi, Θ, c̅ = generate_prediction_model(prob.A, prob.B, prob.Thor; c=prob.c)

    # Define Q̅[i] = blkdg(Q[1][i],..., Q[T][i], P[i])
    Q̅ = [
        BlockDiagonal(vcat([prob.Q[t][i] for t in 1:prob.Thor-1], [prob.P[i]]))
        for i in 1:prob.N
    ]

    # Define R̅ as a block matrix where each block is R̅ᵢⱼ = blkdg(R[1][i][j],..., R[T][i][j])
    R̅ = BlockArray{Float64}(undef_blocks, [prob.Thor * prob.nu[i] for i = 1:prob.N], [prob.Thor * prob.nu[i] for i = 1:prob.N])
    for i in 1:prob.N
        for j in 1:prob.N
            R̅[Block(i, j)] = BlockDiagonal([prob.R[t][i][j] for t=1:prob.Thor])
        end
    end

    # Define H (linear part of the VI mapping)
    H = BlockArray{Float64}(undef_blocks, [prob.Thor * prob.nu[i] for i = 1:prob.N], [prob.Thor * prob.nu[i] for i = 1:prob.N])
    for i in 1:prob.N
        for j in 1:prob.N
            H[Block(i, j)] = Γi[i]' * Q̅[i] * Γi[j]
        end
    end
    H = H + R̅

    # Define F (maps from x0 to the affine part of the VI mapping)
    F = vcat([Γi[i]' * Q̅[i] * Θ for i in 1:prob.N]...)

    # define f (affine part of the VI mapping)
    q̅ = [vcat([prob.q[t][i] for t in 1:prob.Thor-1]..., prob.p[i]) for i in 1:prob.N]
    r̅ = [vcat([prob.r[t][i] for t in 1:prob.Thor]...) for i in 1:prob.N]
    f = vcat([Γi[i]' * (Q̅[i] * c̅ + q̅[i]) + r̅[i] for i in 1:prob.N]...)

    ## Constraints
    # Cᵗₓ*x[t] ≤ bᵗₓ ∀ t ==> C̅ₓΓu̅ <= b̅ₓ -C̅ₓΘx₀ - C̅ₓc̅
    # where C̅ₓ = blkdiag(C¹ₓ, ...,Cᵗₓ)
    C̅_x = BlockDiagonal(prob.C_x)
    # Collect both shared input constraints and state constraints:
    # D_shar = row_stack ([I ⊗ Cu[i]; 
    #                     (I ⊗ C̅ₓ)*Γ[i] ] )
    D_shar = hcat([[BlockDiagonal([prob.C_u[t][i] for t in 1:prob.Thor]);
        C̅_x * Γi[i]] for i in 1:prob.N]...)
    # Append local constraints 
    D = [D_shar;
        BlockDiagonal([ BlockDiagonal([prob.C_loc[t][i] for t in 1:prob.Thor] ) for i = 1:prob.N])]

    # Define E (maps from x0 to constraints)
    E = [zeros(prob.Thor * prob.m_u, prob.nx)        # Shared input constraints
        -1 * C̅_x * Θ;                            # State constraints
        zeros(sum(prob.m_loc) * prob.Thor, prob.nx)] # Local input constraints

    # Affine part of the constraints
    d = [vcat(prob.b_u...);            # Shared input constraints
        vcat(prob.b_x...) - C̅_x * c̅    # State constraints
        vcat([vcat([prob.b_loc[t][i] for t in 1:prob.Thor]...) for i in 1:prob.N]...)] # Local input constraints
    # VI(H*x + F*x0 + f, D*x <= E*x0 + d)
    return mpAVI(Matrix{Float64}(H), F, f, D, E, d)
end

@doc raw"""
    NabetaniParametrization(game::StaticGNEP; θub=nothing, θlb=nothing)
 
Assemble static GNE game into multi-parametric variational inequality (mpAVI).
 
The Nabetani-Tseng-Fukushima reparametrization transforms shared constraints into parameter-dependent bounds:
- Agent i (i<N): ``A_{\text{sh},i} x_i \leq \theta_i``
- Agent N: ``A_{\text{sh},N} x_N \leq b_{\text{sh}} - \sum_{i<N} \theta_i``
 
Returns: ``\text{VI}(H x + f, A x \leq B \theta + b)`` where ``\theta \in [\text{lb}, \text{ub}]``
"""
function NabetaniParametrization(game::StaticGNEP; θub::Union{Vector{Float64},Nothing}=nothing, θlb::Union{Vector{Float64},Nothing}=nothing)
    # Infer dimensions
    N = game.N
    n = game.n
    n_total = sum(n)
    m_sh = length(game.b_sh)
    n_theta = (N - 1) * m_sh

    # Check size of bounds
    if !isnothing(θub) 
        @assert length(θub)==(game.N-1) * m_sh "# of par. upper bounds is $((game.N-1) * m_sh), got $(length(θub))"
    end
    if !isnothing(θlb)
        @assert length(θlb)==(game.N-1) * m_sh "# of par. low bounds is $((game.N-1) * m_sh), got $(length(θlb)) "
    end
    
    # Assemble Hessian (H) from Q blocks 
    H = BlockArray{Float64}(undef_blocks, n, n)
    for i in 1:N
        for j in 1:N
            H[Block(i, j)] = game.Q[i][j]
        end
    end
    H = Matrix(H)
    
    # Assemble linear cost (f) from q vectors
    f = vcat(game.q...)
    
    # Assemble local constraints
    A_loc = BlockDiagonal(game.A_loc)
    b_loc = vcat(game.b_loc...)

    # Assemble Nabetani reparametrization
    # A_hat = blkdiag(A_sh[1], A_sh[2], ..., A_sh[N])
    A_hat_blocks = [game.A_sh[i] for i in 1:N]
    A_hat = BlockDiagonal(A_hat_blocks)
    A_hat = Matrix(A_hat)
    
    # B_g structure (correct for N ≥ 2):
    # Shape: (N*m_sh) × ((N-1)*m_sh)
    # Top block: I_{(N-1)*m_sh}    (agents 1,...,N-1 get explicit allocation θ)
    # Bottom block: -ones(m_sh, (N-1)*m_sh)  (agent N gets remainder)
    B_g_top = I((N - 1) * m_sh)
    B_g_bottom = kron(-1 .* ones(1, N-1), Matrix(I, m_sh, m_sh))
    B_g = vcat(B_g_top, B_g_bottom)
    
    # d_g = [zeros((N-1)*m_sh); b_sh]
    d_g = vcat(zeros((N - 1) * m_sh), game.b_sh)
    
    # Stack all constraints 
    A = vcat(A_loc, A_hat)
    
    # B matrix: local constraints have no theta dependence (zeros), shared constraints have B_g
    B_loc = zeros(size(A_loc, 1), n_theta)
    B = vcat(B_loc, B_g)
    
    b = vcat(b_loc, d_g)
    
    # Return mpAVI
    return mpAVI(H, zeros(n_total, n_theta), f, A, B, b, ub=θub, lb=θlb)
end
####### END Type conversion functions #######

####### Helper functions for optimal GNE selection ##########
@doc raw"""
    filter_gne_crs!(sol, game)

Superseded by [`gne_regions`](@ref) and the equal-slack selection of [`select_optimal_gne`](@ref).

Keeps only the critical regions whose active set contains either all or none of the
copies of each shared constraint. ParametricDAQP returns only a full-row-rank subset of
the active set (e.g. rows of `A_sh` that are zero for a non-incident agent are never
reported), so this test can reject regions that do contain GNEs. It is kept for reference
and is no longer called by [`NabetaniParametrizationSolver`](@ref).
"""
function filter_gne_crs!(sol::ParametricDAQP.Solution, game::StaticGNEP)
    N = game.N
    m_sh = length(game.b_sh)
    n_local = sum(size(game.A_loc[i], 1) for i in 1:N)
    shared_start = n_local + 1
    
    filter!(sol.CRs) do cr
        for i in 1:m_sh
            # Collect all rows of the reformulated constraints associated to the same shared constraints
            player_row_indices = [shared_start + (p - 1) * m_sh + (i - 1) for p in 1:N]
            # Check if they are all active / inactive
            active_status = [row_idx ∈ cr.AS for row_idx in player_row_indices]
            all_active = all(active_status)
            none_active = !any(active_status)
            # If they are neither all active or inactive, the region does not correspond to a set of GNEs
            if !(all_active || none_active)
                return false
            end
        end
        return true
    end
    return length(sol.CRs)
end

@doc raw"""
    equal_slack_map(game::StaticGNEP) -> (L, l)

Affine equal-slack allocation ``\Lambda(x) = Lx + l`` (Nabetani, Tseng, Fukushima, Comput. Optim.
Appl. 48, 2011, Thm. 3.5). With the parametrization of [`NabetaniParametrization`](@ref)
(agent ``i<N``: ``A_{\text{sh},i} x_i \le \theta_i``; agent ``N``:
``A_{\text{sh},N} x_N \le b_{\text{sh}} - \sum_{i<N} \theta_i``), every agent receives its own
consumption plus an equal share ``s/N`` of the slack ``s = b_{\text{sh}} - \sum_j A_{\text{sh},j} x_j``:
```math
\Lambda_i(x) = A_{\text{sh},i} x_i + \tfrac1N \Big(b_{\text{sh}} - \sum_{j} A_{\text{sh},j} x_j\Big), \quad i = 1,\dots,N-1.
```
"""
function equal_slack_map(game::StaticGNEP)
    N, m = game.N, length(game.b_sh)
    offs = cumsum([0; game.n])
    Ā = hcat(game.A_sh...)                        # m × n_total, Σⱼ A_sh,j x_j = Ā x
    L = zeros((N - 1) * m, sum(game.n))
    for i in 1:N-1
        rows = (i-1)*m+1:i*m
        L[rows, offs[i]+1:offs[i+1]] .= game.A_sh[i]
        L[rows, :] .-= Ā ./ N
    end
    return L, repeat(game.b_sh ./ N, N - 1)
end

@doc raw"""
    equal_slack_allocation(game::StaticGNEP, x) -> θ

Equal-slack parameter ``\theta = \Lambda(x)`` of a joint action `x` (see [`equal_slack_map`](@ref)).
If `x` is a GNE, then `x` solves the Nabetani pVI at this θ.
"""
function equal_slack_allocation(game::StaticGNEP, x::AbstractVector)
    L, l = equal_slack_map(game)
    return L * x + l
end

@doc raw"""
    equal_slack_constraint(game::StaticGNEP, K, h) -> (E, e)

Equal-slack equality ``\theta = \Lambda(K\theta + h)`` of a critical region with affine law
``x = K\theta + h``, written as ``E\theta = e`` with ``E = I - LK`` and ``e = Lh + l``
(see [`equal_slack_map`](@ref)).
"""
function equal_slack_constraint(game::StaticGNEP, K::AbstractMatrix, h::AbstractVector)
    L, l = equal_slack_map(game)
    return Matrix(I, size(K, 2), size(K, 2)) - L * K, L * h + l
end

# Replace Eθ = e by an equivalent system with linearly independent rows (the equal-slack
# equality is rank deficient whenever shared rows are slack or zero for some agent);
# returns (nothing, nothing) if the system is inconsistent.
function reduce_equality(E::AbstractMatrix, e::AbstractVector; tol::Float64=1e-9)
    isempty(e) && return E, e
    F = svd(E; full=true)
    r = count(F.S .> tol * max(1.0, maximum(F.S; init=0.0)))
    Ur = F.U[:, 1:r]
    norm(e - Ur * (Ur' * e), Inf) > 1e3 * tol * max(1.0, norm(e, Inf)) && return nothing, nothing
    return Diagonal(F.S[1:r]) * F.Vt[1:r, :], Ur' * e
end

@doc raw"""
    gne_regions(sol, game, mpavi) -> Vector{NamedTuple}

For each critical region ``r`` of the parametric solution `sol` of the Nabetani mpAVI
`mpavi`, return the data defining the equal-slack set
```math
\mathcal E_r = \{\theta \mid G\theta \le g,\ A(K\theta+h) \le B\theta + b,\ \text{lb}\le\theta\le\text{ub},\ E\theta = e\},
```
whose image under ``x = K\theta + h`` is a subset of the GNE set; the union over ``r`` is the
whole GNE set within the parameter box: a pVI solution ``x = K_r\theta + h_r`` is a GNE if and only
if its parameter is the equal-slack allocation ``\theta = \Lambda(x)`` (see [`equal_slack_map`](@ref)),
which follows from Nabetani et al. (2011), Thms. 3.5–3.6. No active-set based region filtering is
needed, which matters because ParametricDAQP returns only a linearly independent subset of the
active constraints. Fields: `K, h` (law in the original θ coordinates, recomputed from the active
set via the KKT system), `G, g` (region polytope in original θ coordinates), `E, e`
(equal-slack equality, reduced to linearly independent rows), `AS`, `region` (index in
`sol.CRs`). Regions whose equality is inconsistent (``\mathcal E_r = \varnothing``) are omitted.
"""
function gne_regions(sol::ParametricDAQP.Solution, game::StaticGNEP, mpavi::mpAVI)
    n = size(mpavi.H, 1)
    regions = NamedTuple[]
    for (k, cr) in enumerate(sol.CRs)
        # Region polytope Athᵀ θ̃ ≤ bth in normalized coordinates θ̃ = (θ - translation) .* scaling
        G = cr.Ath' * Diagonal(sol.scaling)
        g = cr.bth + G * sol.translation
        # Exact law from the (full-row-rank) active set returned by ParametricDAQP.
        # The stored law cr.z is in normalized coordinates and can be inaccurate on
        # badly scaled problems, so it is not used.
        AS = Int.(collect(cr.AS))
        Aa = mpavi.A[AS, :]
        nA = length(AS)
        KKT = [mpavi.H Aa'; Aa zeros(nA, nA)]
        M = KKT \ [-mpavi.F; mpavi.B[AS, :]]
        c = KKT \ [-mpavi.f; mpavi.b[AS]]
        K, h = M[1:n, :], c[1:n]
        E, e = reduce_equality(equal_slack_constraint(game, K, h)...)
        isnothing(E) && continue                     # inconsistent equality: E_r = ∅
        push!(regions, (K=K, h=h, G=G, g=g, E=E, e=e, AS=AS, region=k))
    end
    return regions
end

@doc raw"""
    evaluate_gne(regions, θ; tol=1e-8) -> Vector or nothing

Return ``x = K_r\theta + h_r`` if ``\theta \in \mathcal E_r`` (up to `tol`) for some region of
`regions = gne_regions(sol, game, mpavi)`, i.e. if θ is the equal-slack parameter of a GNE;
`nothing` otherwise. Unlike [`evaluatePWA`](@ref), which returns the pVI solution for any
feasible θ, this only returns GNEs.
"""
function evaluate_gne(regions::AbstractVector, θ::AbstractVector; tol::Float64=1e-8)
    for R in regions
        if all(R.G * θ .<= R.g .+ tol) && (isempty(R.e) || norm(R.E * θ - R.e, Inf) <= tol)
            return R.K * θ + R.h
        end
    end
    return nothing
end

@doc raw"""
    is_gne(game::StaticGNEP, x; tol=1e-7) -> (ok, gap, gaps, infeas)

Certify that `x` is a GNE of `game`, independently of the Nabetani parametrization: `x` must
satisfy the local and shared constraints (violation `infeas`), and for every agent ``i`` the
best response to ``x_{-i}`` under the *original* constraints,
```math
\min_{y}\ \tfrac12 y^\top Q_{ii} y + y^\top\Big(q_i + \sum_{j\ne i} Q_{ij} x_j\Big)
\quad \text{s.t.}\ A_{\text{loc},i}\, y \le b_{\text{loc},i},\quad
A_{\text{sh},i}\, y \le b_{\text{sh}} - \sum_{j\ne i} A_{\text{sh},j} x_j ,
```
must not improve agent ``i``'s cost by more than `tol` (relative to ``\max(1, |J_i(x)|)``). The
right-hand sides are relaxed where needed so that ``x_i`` is feasible for its own best-response
problem; constraint violations of `x` are reported in `infeas` and also checked against `tol`.
`gaps[i]` is the cost improvement of the best response and `gap = maximum(gaps)`.
"""
function is_gne(game::StaticGNEP, x::AbstractVector; tol::Float64=1e-7)
    N = game.N
    offs = cumsum([0; game.n])
    blk(i) = x[offs[i]+1:offs[i+1]]
    Āx = sum(game.A_sh[j] * blk(j) for j in 1:N)
    infeas = max(0.0, maximum(Āx - game.b_sh; init=-Inf),
                 maximum((maximum(game.A_loc[i] * blk(i) - game.b_loc[i]; init=-Inf) for i in 1:N); init=-Inf))
    gaps = zeros(N)
    ok = infeas <= tol
    for i in 1:N
        Qii = game.Q[i][i]
        c = game.q[i] + sum((game.Q[i][j] * blk(j) for j in 1:N if j != i); init=zeros(game.n[i]))
        Ji(y) = 0.5 * y' * Qii * y + c' * y
        model = Model(Clarabel.Optimizer)
        set_silent(model)
        for att in ("tol_gap_abs", "tol_gap_rel", "tol_feas")
            set_attribute(model, att, 1e-10)
        end
        @variable(model, y[1:game.n[i]])
        # right-hand sides are relaxed to contain xᵢ itself, so that a slightly infeasible x
        # (reported separately in `infeas`) still yields a well-posed best-response problem
        size(game.A_loc[i], 1) > 0 && @constraint(model, game.A_loc[i] * y .<= max.(game.b_loc[i], game.A_loc[i] * blk(i)))
        @constraint(model, game.A_sh[i] * y .<= max.(game.b_sh - (Āx - game.A_sh[i] * blk(i)), game.A_sh[i] * blk(i)))
        @objective(model, Min, 0.5 * y' * Qii * y + c' * y)
        optimize!(model)
        if is_solved_and_feasible(model; allow_almost=true)
            gaps[i] = Ji(blk(i)) - Ji(value.(y))
            ok &= gaps[i] <= tol * max(1.0, abs(Ji(blk(i))))
        else
            gaps[i] = Inf
            ok = false
        end
    end
    return (ok=ok, gap=maximum(gaps), gaps=gaps, infeas=infeas)
end

@doc raw"""
    in_gne_map(regions, game, x; tol=1e-8) -> (found, region, θ)

Completeness check for a given point: with ``θ = Λ(x)`` the equal-slack allocation of `x`
([`equal_slack_allocation`](@ref)), check that ``θ ∈ \mathcal E_r`` for some region of
`regions = gne_regions(sol, game, mpavi)` and that ``K_rθ + h_r = x``. Holds for every GNE `x`
whose allocation lies inside the parameter box of the map.
"""
function in_gne_map(regions::AbstractVector, game::StaticGNEP, x::AbstractVector; tol::Float64=1e-8)
    θ = equal_slack_allocation(game, x)
    for R in regions
        if all(R.G * θ .<= R.g .+ tol) && (isempty(R.e) || norm(R.E * θ - R.e, Inf) <= tol) &&
           norm(R.K * θ + R.h - x, Inf) <= tol * max(1.0, norm(x, Inf))
            return (found=true, region=R.region, θ=θ)
        end
    end
    return (found=false, region=0, θ=θ)
end

@doc raw"""
    check_gne_map(sol, game, mpavi; tol=1e-7) -> (ok, regions)

Soundness check of a computed map: for every region with a consistent equal-slack equality,
sample ``\mathcal E_r`` at the optima of the linear programs ``\min \pm\theta_k`` over
``\mathcal E_r`` (its extreme points along the coordinate directions) and at their mean, and
certify every ``x = K_r\theta + h_r`` with [`is_gne`](@ref). Returns `ok` and, per region, the
number of certified points, the largest best-response gap, and the dimension of the sampled
GNE image (`0` = a single point).
"""
function check_gne_map(sol::ParametricDAQP.Solution, game::StaticGNEP, mpavi::mpAVI; tol::Float64=1e-7)
    results = NamedTuple[]
    for R in gne_regions(sol, game, mpavi)
        nθ = size(R.K, 2)
        pts = Vector{Vector{Float64}}()
        for k in 1:nθ, sgn in (1.0, -1.0)
            model = Model(Clarabel.Optimizer)
            set_silent(model)
            @variable(model, mpavi.lb[i] <= θ[i=1:nθ] <= mpavi.ub[i])
            @constraint(model, R.G * θ .<= R.g)
            @constraint(model, mpavi.A * (R.K * θ .+ R.h) .<= mpavi.B * θ .+ mpavi.b)
            isempty(R.e) || @constraint(model, R.E * θ .== R.e)
            @objective(model, Min, sgn * θ[k])
            optimize!(model)
            is_solved_and_feasible(model; allow_almost=true) && push!(pts, value.(θ))
        end
        isempty(pts) && continue                                    # E_r = ∅
        push!(pts, sum(pts) / length(pts))
        xs = [R.K * θ + R.h for θ in pts]
        certs = [is_gne(game, x; tol=tol) for x in xs]
        X = reduce(hcat, xs)
        dim = rank(X .- X[:, 1]; atol=1e-7)
        push!(results, (region=R.region, npoints=length(xs), certified=count(c -> c.ok, certs),
                        gap=maximum(c -> c.gap, certs), dim=dim))
    end
    return (ok=all(r -> r.certified == r.npoints, results), regions=results)
end

# Interior-point solutions are only accurate to O(1e-6) in θ. Refine the regional optimum
# on the affine hull of the numerically active constraints of E_r: for a quadratic objective
# by solving the equality-constrained QP exactly (Hessian `P`, gradient `p`), otherwise by
# projecting onto that affine hull. The refinement is kept only if it is feasible and does
# not increase the objective.
function polish_regional_solution(θ::AbstractVector, R, mpavi::mpAVI, ϕ::Function;
                                  P=nothing, p=nothing, act_tol::Float64=1e-6)
    nθ = length(θ)
    Ain = [R.G; mpavi.A * R.K - mpavi.B; Matrix(I, nθ, nθ); -Matrix(I, nθ, nθ)]
    bin = [R.g; mpavi.b - mpavi.A * R.h; mpavi.ub; -mpavi.lb]
    act = findall(bin - Ain * θ .<= act_tol)
    Aeq = [R.E; Ain[act, :]]
    beq = [R.e; bin[act]]
    # rows that are zero up to round-off (e.g. allocation rows that are identically active on
    # the region) carry no information and would spoil the rank decisions below
    keep = [norm(Aeq[k, :], Inf) > 1e-10 for k in 1:size(Aeq, 1)]
    Aeq, beq = Aeq[keep, :], beq[keep]
    if isnothing(P)
        isempty(beq) && return θ
        θs = θ - pinv(Aeq) * (Aeq * θ - beq)
    else
        # Null-space solve of min ½θᵀPθ + pᵀθ s.t. Aeq θ = beq (handles redundant rows)
        θ0 = isempty(beq) ? θ : θ - pinv(Aeq) * (Aeq * θ - beq)
        Z = isempty(beq) ? Matrix(I, nθ, nθ) : nullspace(Aeq)
        θs = size(Z, 2) == 0 ? θ0 : θ0 - Z * (pinv(Z' * P * Z) * (Z' * (P * θ0 + p)))
    end
    feasible = all(Ain * θs .<= bin .+ 1e-9) && (isempty(R.e) || norm(R.E * θs - R.e, Inf) <= 1e-9)
    return feasible && ϕ(R.K * θs + R.h) <= ϕ(R.K * θ + R.h) + 1e-12 ? θs : θ
end

# Hessian and gradient of a quadratic JuMP objective in the variables `θ`
function quadratic_coefficients(obj, θ::AbstractVector)
    idx = Dict(v => i for (i, v) in enumerate(θ))
    nθ = length(θ)
    P, p = zeros(nθ, nθ), zeros(nθ)
    q = obj isa QuadExpr ? obj : convert(QuadExpr, obj)
    for (pair, c) in q.terms
        i, j = idx[pair.a], idx[pair.b]
        if i == j
            P[i, i] += 2c
        else
            P[i, j] += c; P[j, i] += c
        end
    end
    for (v, c) in q.aff.terms
        p[idx[v]] += c
    end
    return P, p
end

@doc raw"""
    select_optimal_gne(sol, game, mpavi, ϕ, is_quadratic, optimizer=Clarabel.Optimizer; verify=false)

Algorithm 1 (optimal GNE selection): for every critical region solve
```math
\min_\theta\ \phi(K_r\theta + h_r) \quad \text{s.t.}\ \theta \in \mathcal E_r
```
(see [`gne_regions`](@ref)), skip regions with empty ``\mathcal E_r``, and return the best
regional solution. `θ_star` is returned in the original parameter coordinates. With
`verify=true` the selected point is certified with [`is_gne`](@ref).
"""
function select_optimal_gne(
    sol::ParametricDAQP.Solution,
    game::StaticGNEP,
    mpavi::mpAVI,
    ϕ::Function,
    is_quadratic::Bool,
    optimizer = Clarabel.Optimizer;
    verify::Bool = false
)::OptimalGNEResult
    if !is_quadratic && optimizer == Clarabel.Optimizer
        optimizer = Ipopt.Optimizer
    end
    candidates = []
    for R in gne_regions(sol, game, mpavi)
        n_θ = size(R.K, 2)
        n_u = size(R.K, 1)
        model = Model(optimizer)
        set_silent(model)
        if optimizer == Clarabel.Optimizer
            # The equal-slack sets are typically lower dimensional: solve to high accuracy
            for att in ("tol_gap_abs", "tol_gap_rel", "tol_feas")
                set_attribute(model, att, 1e-10)
            end
        end
        @variable(model, mpavi.lb[i] <= θ[i=1:n_θ] <= mpavi.ub[i])
        @constraint(model, R.G * θ .<= R.g)
        @constraint(model, mpavi.A * (R.K * θ .+ R.h) .<= mpavi.B * θ .+ mpavi.b)
        @constraint(model, R.E * θ .== R.e)
        if is_quadratic
            @objective(model, Min, ϕ(R.K * θ .+ R.h))
        else
            @variable(model, y[1:n_u])
            @constraint(model, y .== R.K * θ .+ R.h)
            @operator(model, op_ϕ, n_u, (ys...) -> ϕ(collect(ys)))
            @objective(model, Min, op_ϕ(y...))
        end
        optimize!(model)
        if is_solved_and_feasible(model)
            if is_quadratic
                P, p = quadratic_coefficients(objective_function(model), θ)
                θ_opt = polish_regional_solution(value.(θ), R, mpavi, ϕ; P=P, p=p)
            else
                θ_opt = polish_regional_solution(value.(θ), R, mpavi, ϕ)
            end
            u_opt = R.K * θ_opt + R.h
            push!(candidates, (θ=θ_opt, u=u_opt, φ=ϕ(u_opt), region=R.region))
        end
        # Infeasible regional problems (E_r = ∅) are the normal case and are skipped
    end
    isempty(candidates) && error("[select_optimal_gne] No region with a nonempty equal-slack set: no GNE in the parameter box.")
    best = candidates[argmin([c.φ for c in candidates])]
    if verify
        cert = is_gne(game, best.u)
        cert.ok || @warn "[select_optimal_gne] selected point failed the GNE certificate" cert.gap cert.infeas
    end
    return OptimalGNEResult(best.θ, best.u, best.φ, best.region, candidates)
end

####### END Helper functions for optimal GNE selection ##########

####### Dynamics functions #######

@doc raw"""
    generate_prediction_model(A, Bi, T_hor)

Constructs the prediction model matrices, ``\Gamma_i``, and ``\Theta``, and a vector ``\bar{c}`` for a discrete-time affine system
```math 
x^+ = Ax + \sum_{i=1}^N (B_i u_i) + c
```
Prediction model:
```math
\bar{x} = \Theta x_0 + \sum_{i=1}^N (\Gamma_i\bar{u}_i) + \bar{c}
```
where
```math
\bar{x}=\text{col}(x[1], ... , x[T]), \quad \bar{u}_i = \text{col}(u_i[0], ..., u_i[T-1])
```
Specifically, the matrices are
```math
\Gamma_i = \begin{bmatrix}
	B_i & 0 & 0 & ... & 0 \\
	AB_i & B_i & 0 & ... & 0 \\
	\vdots & & \ddots & & \\
	A^{T-1}B_i & A^{T-2}B_i & ... &  & B_i
	\end{bmatrix} \quad \Theta = \begin{bmatrix}
		A \\ A^2 \\ ... \\ A^T
	\end{bmatrix} \quad \bar{c} = \begin{bmatrix}
	I & 0 & 0 & ... & 0 \\
	A & I & 0 & ... & 0 \\
	\vdots & & \ddots & & \\
	A^{T-1} & A^{T-2} & ... &  & I
	\end{bmatrix} (1_{T} \otimes c)
```
# Arguments
- `A`: State transition matrix.
- `B`: Vector of input matrices for each agent or subsystem.
- `T_hor`: Prediction horizon (number of time steps).
- `c`: affine part of the dynamics. If omitted, defaults to a vector of zeros.

# Returns
- `Γ`: Matrix mapping the stacked input sequences to the stacked state sequence.
- `Γi`: Vector of views into `Γ`, each corresponding to the input sequence of an agent.
- `Θ`: Matrix mapping the initial state to the stacked state sequence.
- `c̅`: Affine additive vector.
"""
function generate_prediction_model(A::Matrix{Float64}, B::Vector{Matrix{Float64}}, T_hor::Int; c::Vector{Float64}=zeros(size(A, 1)))
    # system: x⁺=Ax + ∑(Bᵢuᵢ) + c
    # prediction model: x̅ = Θx₀+(∑ Γᵢu̅ᵢ) + c̅
    n_x = size(A, 1)
    n_u = map(x -> size(x, 2), B)
    N = length(B)
    Γ = zeros(n_x * T_hor, sum(n_u) * T_hor) # Maps the column stack of input sequences to state sequence
    Γi = Vector{SubArray}(undef, N) # Views into Γ
    start_col = 1
    for i in 1:N
        Γi[i] = @view Γ[:, start_col:start_col+n_u[i]*T_hor-1]
        start_col += n_u[i] * T_hor
    end
    # Define the Γ matrices
    for i in 1:N
        Γi[i][1:n_x, 1:n_u[i]] .= B[i]
        for t in 1:T_hor-1
            current_rows = n_x*t+1:n_x*(t+1)
            previous_rows = n_x*(t-1)+1:n_x*t
            Γi[i][current_rows, n_u[i]+1:end] .= Γi[i][previous_rows, 1:end-n_u[i]] # Shift the previous block row by n_u
            Γi[i][current_rows, 1:n_u[i]] .= A * Γi[i][previous_rows, 1:n_u[i]] # Multiply the leftmost block of the previous row by A
        end
    end

    Θ = vcat([A^t for t in 1:T_hor]...) # Maps the initial state to state sequence

    # Affine part
    Γc = zeros(n_x * T_hor, n_x * T_hor)
    Γc[1:n_x, 1:n_x] .= Matrix{Float64}(I(n_x))
    for t in 1:T_hor-1
        current_rows = n_x*t+1:n_x*(t+1)
        previous_rows = n_x*(t-1)+1:n_x*t
        Γc[current_rows, n_x+1:end] .= Γc[previous_rows, 1:end-n_x] # Shift the previous block row by n_x
        Γc[current_rows, 1:n_x] .= A * Γc[previous_rows, 1:n_x] # Multiply the leftmost block of the previous row by A
    end
    c̅ = Γc * kron(ones(T_hor), c)

    return Γ, Γi, Θ, c̅
end

# For time-varying systems
function generate_prediction_model(A::Vector{Matrix{Float64}}, 
        B::Vector{Vector{Matrix{Float64}}},     
        T_hor::Int; 
        c::Vector{Vector{Float64}}=[zeros(size(A[1], 1)) for _ in 1:T_hor])
    # system: x⁺=Aᵗx + ∑(Bᵢᵗuᵢ) + cᵗ
    # prediction model: x̅ = Θx₀+(∑ Γᵢu̅ᵢ) + c̅
    n_x = size(A[1], 1)
    n_u = map(x -> size(x, 2), B[1])
    N = length(B[1])
    Γ = zeros(n_x * T_hor, sum(n_u) * T_hor) # Maps the column stack of input sequences to state sequence
    Γi = Vector{SubArray}(undef, N) # Views into Γ
    start_col = 1
    for i in 1:N
        Γi[i] = @view Γ[:, start_col:start_col+n_u[i]*T_hor-1]
        start_col += n_u[i] * T_hor
    end
    # Define the Γ matrices
    # Γᵢ = [ Bᵢ¹      0    0  ... 0;
    #       A²Bᵢ¹    Bᵢ²   0  ... 0;
    #       A³A²Bᵢ¹  A³Bᵢ² Bᵢ³ ... 0; 
    # ...
    #]
    for i in 1:N
        Γi[i][1:n_x, 1:n_u[i]] .= B[1][i] # Place Bᵢ[1] on the top left block
        for t in 2:T_hor
            current_rows = n_x*(t-1)+1:n_x*(t)
            previous_rows = n_x*(t-2)+1:n_x*(t-1)
            Γi[i][current_rows, :] .= Γi[i][previous_rows, :] # copy the previous block row to the current one
            Γi[i][current_rows, :] .= A[t] * Γi[i][current_rows, :] # Premultiply by A[t]
            Γi[i][current_rows, n_u[i]*(t-1)+1:n_u[i]*t] .= B[t][i] # Place Bᵢ[t] on the diagonal block
        end
    end
    # Define the Θ matrix (multiplying x_0)
    # Θ = [ A¹    ;
    #       A²A¹  ;
    #       A³A²A¹; 
    #      ...]
    Θ = zeros(n_x * T_hor, n_x)
    Θ[1:n_x, :] = A[1]
    for t in 2:T_hor
        current_rows = n_x*(t-1)+1:n_x*(t)
        previous_rows = n_x*(t-2)+1:n_x*(t-1)
        Θ[current_rows,:] = A[t] * Θ[previous_rows,:]
    end

    # Affine part
    # Define the Γ_c  matrix
    # Γ_c = [ I     0     0 ... 0;
    #       A²    I     0 ... 0;
    #       A³A²  A³A²  I ... 0; 
    # ...]
    # Then, the affine part of the prediction model will be
    # c̄ = Γ_c * [c₁; c₂, ....]

    Γc = zeros(n_x * T_hor, n_x * T_hor)
    Γc[1:n_x, 1:n_x] .= Matrix{Float64}(I(n_x)) # Place I on the top left block
    for t in 2:T_hor
        current_rows = n_x*(t-1)+1:n_x*(t)
        previous_rows = n_x*(t-2)+1:n_x*(t-1)
        Γc[current_rows, :] .= Γc[previous_rows, :] # copy the previous block row to the current one
        Γc[current_rows, :] .= A[t] * Γc[current_rows, :] # Premultiply by A[t]
        Γc[current_rows, n_x*(t-1)+1:n_x*t] .= Matrix{Float64}(I(n_x)) # Place I on the diagonal block
    end
    c̅ = Γc * vcat(c...)

    return Γ, Γi, Θ, c̅
end


"""
    is_stabilizable(A, B) -> Bool

Return `true` if the pair `(A, B)` is stabilizable, i.e. every unstable mode of `A`
(eigenvalue with |λ| ≥ 1) is controllable. Uses the Hautus lemma.
"""
function is_stabilizable(A, B)
    eigv_A = eigvals(A)
    n_x = size(A, 1)
    for λ in eigv_A
        if abs(λ) ≥ 1
            M = hcat(λ * I(n_x) - A, B)
            if rank(M) < n_x
                return false
            end
        end
    end
    return true
end

"""
    is_detectable(A, C) -> Bool

Return `true` if the pair `(A, C)` is detectable, i.e. every unstable mode of `A`
(eigenvalue with |λ| ≥ 1) is observable. Uses the Hautus lemma.
"""
function is_detectable(A, C)
    eigv_A = eigvals(A)
    n_x = size(A, 1)
    for λ in eigv_A
        if abs(λ) ≥ 1
            M = vcat(λ * I(n_x) - A, C)
            if rank(M) < n_x
                return false
            end
        end
    end
    return true
end
####### END Dynamics functions #######

"""
    compute_residual(prob::AVI, x::AbstractVector) -> Float64

Compute the natural residual of the AVI at point `x`: the distance from `x` to its
projection onto the feasible set along the VI mapping direction.
"""
function compute_residual(prob::AVI, x::AbstractVector)
    y = x - (prob.H * x + prob.f)
    proj = DAQP.Model()
    DAQP.setup(proj, Matrix{Float64}(I, prob.n, prob.n), -y, Matrix{Float64}(prob.A), prob.b, Float64[], zeros(Cint, prob.m))
    x_transf, _, _, _ = DAQP.solve(proj)
    return norm(x - x_transf)
end

@doc raw"""
    setParameterSpace(mpavi::mpAVI; C, d, ub, lb) -> mpAVI

Return a new `mpAVI` with updated parameter-space bounds. Any argument left as `nothing`
retains the value from `mpavi`.

# Keyword Arguments
- `C`, `d`: Polytope constraint matrix and right-hand side (`Cθ ≤ d`).
- `ub`, `lb`: Upper and lower box bounds on the parameter `θ`.
"""
function setParameterSpace(mpavi::mpAVI;
    C::Union{AbstractMatrix{Float64},Nothing}=nothing,
    d::Union{AbstractVector{Float64},Nothing}=nothing,
    ub::Union{AbstractVector{Float64},Nothing}=nothing,
    lb::Union{AbstractVector{Float64},Nothing}=nothing)
    C = isnothing(C) ? mpavi.C : C
    d = isnothing(d) ? mpavi.d : d
    ub = isnothing(ub) ? mpavi.ub : ub
    lb = isnothing(lb) ? mpavi.lb : lb
    return mpAVI(mpavi.H, mpavi.F, mpavi.f, mpavi.A, mpavi.B, mpavi.b; C, d, ub, lb)
end


# Computes the hessian of a multi-input function
function block_hessian(f, x...)
    # Vectorize x
    v = vcat(x...)
    # Create selection ranges on v that correspond to the elements of x
    lengths = length.(x)
    offsets = cumsum((1, lengths[1:end-1]...))
    ranges = [offsets[i] : offsets[i] + lengths[i] - 1 for i in eachindex(x)]

    # Vectorize f 
    f_vec(v) = f(ntuple(i -> # The ntuple is better understood by Zygote than generators, apparently
        @view(v[offsets[i]:offsets[i] + lengths[i] - 1]),
        length(x)
    )...)

    H = Zygote.hessian(f_vec, v)

    blocks = Dict(
    (i, j) => @view H[
        ranges[i],
        ranges[j]
    ]
    for i in eachindex(x),
        j in eachindex(x)
    )
    return H, blocks
end