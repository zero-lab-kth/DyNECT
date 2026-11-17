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

**Deprecated legacy routine**, kept only for comparisons; not used by the default pipeline.

Keeps the critical regions whose active set contains either all or none of the copies of each
shared constraint (the Nabetani condition evaluated on `cr.AS`). ParametricDAQP returns only a
*linearly independent* active-set basis, which need not contain every binding constraint (zero
rows of a non-incident agent, linearly dependent binding rows), so membership in `cr.AS` is not
equivalent to activity and this test can drop regions that contain GNEs. Use
[`gne_regions`](@ref), which certifies regions through the equal-slack equality instead.
"""
function filter_gne_crs!(sol::ParametricDAQP.Solution, game::StaticGNEP)
    Base.depwarn("filter_gne_crs! evaluates the Nabetani condition on the independent active-set basis and can drop GNE regions; use gne_regions (equal-slack certification) instead.", :filter_gne_crs!)
    N = game.N
    m_sh = length(game.b_sh)
    n_local = sum(size(game.A_loc[i], 1) for i in 1:N)
    shared_start = n_local + 1

    filter!(sol.CRs) do cr
        for i in 1:m_sh
            player_row_indices = [shared_start + (p - 1) * m_sh + (i - 1) for p in 1:N]
            active_status = [row_idx ∈ cr.AS for row_idx in player_row_indices]
            if !(all(active_status) || !any(active_status))
                return false
            end
        end
        return true
    end
    return length(sol.CRs)
end

@doc raw"""
    equal_slack_map(game::StaticGNEP; Bγ = zeros(m, 0)) -> (L, Lγ, l)

Equal-slack allocation (Nabetani, Tseng, Fukushima, Comput. Optim. Appl. 48, 2011, Thm. 3.5)
in DyNECT's allocation convention. With shared constraints ``\sum_j A_{\text{sh},j} x_j \le
b_{\text{sh}} + B_\gamma\gamma`` (``B_\gamma`` empty without a leader), the Nabetani
parametrization gives agent ``i<N`` the allocation ``\beta_i = \theta_i`` and agent ``N`` the
remainder ``\beta_N = b_{\text{sh}} + B_\gamma\gamma - \sum_{i<N}\theta_i``, so that
```math
\sum_{i=1}^N \beta_i = b_{\text{sh}} + B_\gamma\gamma .
```
The equal-slack allocation gives every agent its own consumption plus an equal share of the slack:
```math
\beta_i = A_{\text{sh},i} x_i + \tfrac1N\Big(b_{\text{sh}} + B_\gamma\gamma - \sum_j A_{\text{sh},j} x_j\Big),
\qquad i = 1,\dots,N,
```
and its free part ``\beta_{1:N-1} = Lx + L_\gamma\gamma + l`` is returned, with
``L = \mathrm{blkdiag}(A_{\text{sh},i})_{i<N} - \tfrac1N \mathbf 1_{N-1}\otimes[A_{\text{sh},1}\cdots A_{\text{sh},N}]``
(size ``(N-1)m \times n``), ``L_\gamma = \tfrac1N\mathbf 1_{N-1}\otimes B_\gamma`` (size
``(N-1)m \times n_\gamma``) and ``l = \tfrac1N\mathbf 1_{N-1}\otimes b_{\text{sh}}``.
In the per-agent form ``g_i(x_i,\gamma) = A_{\text{sh},i}x_i - (b_{\text{sh}}+B_\gamma\gamma)/N``,
``\sum_i g_i \le 0``, this is ``\beta_i - (b_{\text{sh}}+B_\gamma\gamma)/N = g_i - \tfrac1N\sum_j g_j``.
"""
function equal_slack_map(game::StaticGNEP; Bγ::AbstractMatrix=zeros(length(game.b_sh), 0))
    N, m = game.N, length(game.b_sh)
    @assert size(Bγ, 1) == m "Bγ must have m = $m rows"
    offs = cumsum([0; game.n])
    Ā = hcat(game.A_sh...)                        # m × n_total, Σⱼ A_sh,j x_j = Ā x
    L = zeros((N - 1) * m, sum(game.n))
    for i in 1:N-1
        rows = (i-1)*m+1:i*m
        L[rows, offs[i]+1:offs[i+1]] .= game.A_sh[i]
        L[rows, :] .-= Ā ./ N
    end
    return L, repeat(Matrix{Float64}(Bγ) ./ N, N - 1), repeat(game.b_sh ./ N, N - 1)
end

@doc raw"""
    equal_slack_allocation(game::StaticGNEP, x; γ = Float64[], Bγ = zeros(m, 0)) -> θ

Free allocations ``\theta = \beta_{1:N-1} = \Lambda(x,\gamma)`` of the equal-slack allocation
of `x` (see [`equal_slack_map`](@ref)). If `x` is a GNE, it solves the Nabetani pVI at this θ
(and, with a leader, at ``p = (\theta, \gamma)``).
"""
function equal_slack_allocation(game::StaticGNEP, x::AbstractVector;
                                γ::AbstractVector=Float64[], Bγ::AbstractMatrix=zeros(length(game.b_sh), 0))
    L, Lγ, l = equal_slack_map(game; Bγ=Bγ)
    return L * x + (isempty(γ) ? zeros(length(l)) : Lγ * γ) + l
end

@doc raw"""
    equal_slack_constraint(game::StaticGNEP, K, h; Bγ = zeros(m, 0)) -> (E, e)

Equal-slack equality of a critical region with law ``x = K p + h`` on the parameter
``p = (\theta, \gamma)``: ``\theta = \Lambda(Kp + h, \gamma)``, i.e.
```math
(P_\beta - L K - L_\gamma P_\gamma)\, p = L h + l ,
```
with ``P_\beta = [I\ 0]`` and ``P_\gamma = [0\ I]`` selecting θ and γ in p.
"""
function equal_slack_constraint(game::StaticGNEP, K::AbstractMatrix, h::AbstractVector;
                                Bγ::AbstractMatrix=zeros(length(game.b_sh), 0))
    L, Lγ, l = equal_slack_map(game; Bγ=Bγ)
    nβ, nγ = size(L, 1), size(Lγ, 2)
    @assert size(K, 2) == nβ + nγ "K must have (N-1)m + n_γ = $(nβ + nγ) columns, got $(size(K, 2))"
    Pβ = [Matrix{Float64}(I, nβ, nβ) zeros(nβ, nγ)]
    Pγ = [zeros(nγ, nβ) Matrix{Float64}(I, nγ, nγ)]
    return Pβ - L * K - Lγ * Pγ, L * h + l
end

# Numerical row reduction of Eθ = e to linearly independent rows (the equal-slack equality is
# rank deficient whenever shared rows are slack or zero for some agent). The rank is decided
# with a tolerance: this is a numerical step, not an exact certificate. Returns
# (nothing, nothing) if the system is numerically inconsistent.
function reduce_equality(E::AbstractMatrix, e::AbstractVector; tol::Float64=1e-9)
    isempty(e) && return E, e
    F = svd(E; full=true)
    r = count(F.S .> tol * max(1.0, maximum(F.S; init=0.0)))
    Ur = F.U[:, 1:r]
    norm(e - Ur * (Ur' * e), Inf) > 1e3 * tol * max(1.0, norm(e, Inf)) && return nothing, nothing
    return Diagonal(F.S[1:r]) * F.Vt[1:r, :], Ur' * e
end

@doc raw"""
    gne_regions(sol, game, mpavi; Bγ = zeros(m, 0), include_inconsistent = false) -> Vector{NamedTuple}

For each critical region ``\mathcal P_r`` of the explicit solution `sol` of the Nabetani mpAVI
`mpavi`, the equal-slack certified set
```math
\mathcal E_r = \{p \mid G_r p \le g_r,\ A(K_r p+h_r) \le Bp + b,\ \text{lb}\le p\le\text{ub},\
(P_\beta - L K_r - L_\gamma P_\gamma)p = L h_r + l\}.
```
A pVI solution ``x = K_r p + h_r`` is a GNE if and only if ``p \in \mathcal E_r`` for some ``r``
(Nabetani et al. 2011, Thms. 3.5–3.6, with the equal-slack allocation of
[`equal_slack_map`](@ref)); no active-set based filtering is used. Assumptions:
- *Soundness* (every ``K_r p + h_r`` with ``p \in \mathcal E_r`` is a GNE) requires that the
  regional law solves the parametrized VI on ``\mathcal P_r`` and that every player's problem is
  convex (symmetric ``Q_{ii} \succeq 0``, linear constraints); it does not require monotonicity
  of the pseudo-gradient.
- *Completeness* (every GNE is represented) additionally requires that the parametrized VI has a
  unique solution at the equal-slack allocation of the GNE (strong monotonicity, i.e. a positive
  definite symmetric part of ``H``, is sufficient but not necessary), that this allocation lies in
  the parameter box, and that the enumeration covers the box (check `sol.status`).
- The equality is reduced to independent rows with a numerical rank decision, so the
  representation is exact only up to that tolerance.

Fields: `K, h` (law in original coordinates, recomputed from the active set via the KKT system),
`G, g` (region polytope in original coordinates), `Ain, bin` (all inequalities of
``\mathcal E_r``: region polytope, primal feasibility ``A(K_r p+h_r) \le Bp + b`` and the
parameter box), `E, e` (equality, numerically reduced to independent rows), `AS`, `region` (index in `sol.CRs`), `consistent` (false if the equality is
numerically inconsistent, i.e. ``\mathcal E_r = \varnothing``; such regions are returned only with
`include_inconsistent = true`).
"""
function gne_regions(sol::ParametricDAQP.Solution, game::StaticGNEP, mpavi::mpAVI;
                     Bγ::AbstractMatrix=zeros(length(game.b_sh), 0), include_inconsistent::Bool=false)
    n = size(mpavi.H, 1)
    λmin = minimum(eigvals(Symmetric((mpavi.H + mpavi.H') / 2)))
    λmin > 0 || @warn "[gne_regions] the symmetric part of H is not positive definite (λ_min = $λmin): this sufficient condition for a unique parametrized-VI solution fails, so completeness of the represented GNE set is not guaranteed (soundness of the certified sets is unaffected)"
    regions = NamedTuple[]
    for (k, cr) in enumerate(sol.CRs)
        # region polytope Athᵀ p̃ ≤ bth in normalized coordinates p̃ = (p - translation) .* scaling
        G = cr.Ath' * Diagonal(sol.scaling)
        g = cr.bth + G * sol.translation
        # law from the (full-row-rank) active set via its KKT system. The stored law cr.z is not
        # used: ParametricDAQP v0.6.1 drops an active constraint whose multiplier is constant in
        # the parameter (normalize_model sets its normalization factor to 0)
        AS = Int.(collect(cr.AS))
        Aa = mpavi.A[AS, :]
        nA = length(AS)
        KKT = [mpavi.H Aa'; Aa zeros(nA, nA)]
        M = KKT \ [-mpavi.F; mpavi.B[AS, :]]
        c = KKT \ [-mpavi.f; mpavi.b[AS]]
        K, h = M[1:n, :], c[1:n]
        # all inequalities of E_r: region polytope, primal feasibility of the law, parameter box
        nθ = size(K, 2)
        Ain = [G; mpavi.A * K - mpavi.B; Matrix(1.0I, nθ, nθ); -Matrix(1.0I, nθ, nθ)]
        bin = [g; mpavi.b - mpavi.A * h; mpavi.ub; -mpavi.lb]
        E, e = reduce_equality(equal_slack_constraint(game, K, h; Bγ=Bγ)...)
        if isnothing(E)
            include_inconsistent && push!(regions, (K=K, h=h, G=G, g=g, Ain=Ain, bin=bin, E=zeros(0, nθ), e=Float64[],
                                                    AS=AS, region=k, consistent=false))
            continue
        end
        push!(regions, (K=K, h=h, G=G, g=g, Ain=Ain, bin=bin, E=E, e=e, AS=AS, region=k, consistent=true))
    end
    return regions
end

@doc raw"""
    evaluate_gne(regions, p; tol=1e-8) -> Vector or nothing

Return ``x = K_r p + h_r`` if ``p \in \mathcal E_r`` (up to `tol`; all inequalities `Ain p ≤ bin`
and the equality, i.e. the same set the selection optimizes over) for some region of
`regions = gne_regions(...)`, i.e. if p is the equal-slack parameter of a GNE; `nothing`
otherwise. Unlike [`evaluatePWA`](@ref), which returns the pVI solution for any feasible p,
this only returns GNEs.
"""
function evaluate_gne(regions::AbstractVector, θ::AbstractVector; tol::Float64=1e-8)
    for R in regions
        get(R, :consistent, true) || continue
        if all(R.Ain * θ .<= R.bin .+ tol) && (isempty(R.e) || norm(R.E * θ - R.e, Inf) <= tol)
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
The cost of player ``i`` is taken as ``\tfrac12 x_i^\top Q_{ii} x_i + \dots``, so ``Q_{ii}`` is
assumed symmetric (as it must be for the pseudo-gradient ``Hx + f`` to describe this game).
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
    in_gne_map(regions, game, x; γ = Float64[], Bγ = zeros(m, 0), tol = 1e-8) -> (found, region, p)

Numerical completeness check for a given point: with ``p = (\Lambda(x,\gamma), \gamma)``
([`equal_slack_allocation`](@ref)), check that ``p \in \mathcal E_r`` for some region of
`regions` and that ``K_r p + h_r = x``. Holds for every GNE `x` whose allocation lies inside the
parameter box of the map, provided the enumeration covers that box.
"""
function in_gne_map(regions::AbstractVector, game::StaticGNEP, x::AbstractVector;
                    γ::AbstractVector=Float64[], Bγ::AbstractMatrix=zeros(length(game.b_sh), 0), tol::Float64=1e-8)
    p = [equal_slack_allocation(game, x; γ=γ, Bγ=Bγ); γ]
    for R in regions
        get(R, :consistent, true) || continue
        if all(R.Ain * p .<= R.bin .+ tol) && (isempty(R.e) || norm(R.E * p - R.e, Inf) <= tol) &&
           norm(R.K * p + R.h - x, Inf) <= tol * max(1.0, norm(x, Inf))
            return (found=true, region=R.region, θ=p)
        end
    end
    return (found=false, region=0, θ=p)
end

@doc raw"""
    check_gne_map(sol, game, mpavi; Bγ = zeros(m, 0), game_at = γ -> game, tol = 1e-7) -> (ok, regions)

Numerical soundness check of a computed map (a verification aid, not part of the construction
of ``\mathcal E_r``): for every region with a consistent equal-slack equality, take the optima of
the linear programs ``\min \pm p_k`` over ``\mathcal E_r`` and their mean, and certify every
``x = K_r p + h_r`` with [`is_gne`](@ref) on `game_at(γ)` (the game at the leader value contained
in p). Returns `ok` and, per region, the number of certified points, the largest best-response
gap and the dimension of the sampled GNE image (`0` = a single point).
"""
function check_gne_map(sol::ParametricDAQP.Solution, game::StaticGNEP, mpavi::mpAVI;
                       Bγ::AbstractMatrix=zeros(length(game.b_sh), 0), game_at::Function=γ -> game,
                       tol::Float64=1e-7)
    nγ = size(Bγ, 2)
    results = NamedTuple[]
    for R in gne_regions(sol, game, mpavi; Bγ=Bγ)
        nθ = size(R.K, 2)
        pts = Vector{Vector{Float64}}()
        for k in 1:nθ, sgn in (1.0, -1.0)
            model = Model(Clarabel.Optimizer)
            set_silent(model)
            @variable(model, θ[1:nθ])
            @constraint(model, R.Ain * θ .<= R.bin)
            isempty(R.e) || @constraint(model, R.E * θ .== R.e)
            @objective(model, Min, sgn * θ[k])
            optimize!(model)
            is_solved_and_feasible(model; allow_almost=true) && push!(pts, value.(θ))
        end
        isempty(pts) && continue
        push!(pts, sum(pts) / length(pts))
        certs = [is_gne(game_at(p[end-nγ+1:end]), R.K * p + R.h; tol=tol) for p in pts]
        X = reduce(hcat, [R.K * p + R.h for p in pts])
        dim = rank(X .- X[:, 1]; atol=1e-7)
        push!(results, (region=R.region, npoints=length(pts), certified=count(c -> c.ok, certs),
                        gap=maximum(c -> c.gap, certs), dim=dim))
    end
    return (ok=all(r -> r.certified == r.npoints, results), regions=results)
end

# Interior-point solutions are only accurate to O(1e-6) in p. Refine the regional optimum on
# the affine hull of the numerically active constraints of E_r: for a quadratic objective by
# solving the equality-constrained QP exactly (Hessian `P`, gradient `q`), otherwise by
# projecting onto that affine hull. The refinement is kept only if it is feasible and either does
# not increase the objective `ϕp(p)` or moves the point by at most a relative 1e-6.
function polish_regional_solution(θ::AbstractVector, R, mpavi::mpAVI, ϕp::Function, Ain, bin;
                                  P=nothing, q=nothing, act_tol::Float64=1e-6)
    nθ = length(θ)
    act = findall(bin - Ain * θ .<= act_tol)
    # keep the equality and add an active row only if it increases the rank: on degenerate faces
    # several active rows are parallel and their right-hand sides agree only up to round-off
    Aeq, beq = Matrix(R.E), copy(R.e)
    for k in act
        norm(Ain[k, :], Inf) > 1e-10 || continue
        A2 = [Aeq; Ain[k:k, :]]
        rank(A2; rtol=1e-9) > size(Aeq, 1) || continue
        Aeq, beq = A2, [beq; bin[k]]
    end
    if isnothing(P)
        isempty(beq) && return θ
        θs = θ - pinv(Aeq) * (Aeq * θ - beq)
    else
        θ0 = isempty(beq) ? θ : θ - pinv(Aeq) * (Aeq * θ - beq)
        Z = isempty(beq) ? Matrix(I, nθ, nθ) : nullspace(Aeq)
        θs = size(Z, 2) == 0 ? θ0 : θ0 - Z * (pinv(Z' * P * Z) * (Z' * (P * θ0 + q)))
    end
    feasible = all(Ain * θs .<= bin .+ 1e-9) && (isempty(R.e) || norm(R.E * θs - R.e, Inf) <= 1e-9)
    # accept if the objective does not increase, or if the move is tiny: the solver point may be
    # slightly infeasible and hence have a slightly lower objective than the feasible refinement
    small = norm(θs - θ, Inf) <= 1e-6 * max(1.0, norm(θ, Inf))
    return feasible && (ϕp(θs) <= ϕp(θ) + 1e-12 || small) ? θs : θ
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

# Classification of a regional solve. Numerical failures are never reported as infeasibility,
# and a solver's report of (local) infeasibility is only provisional (`:infeasible_reported`);
# see `certify_empty`.
function regional_status(model)
    st = termination_status(model)
    st in (OPTIMAL, LOCALLY_SOLVED) && primal_status(model) == FEASIBLE_POINT && return :solved
    st in (ALMOST_OPTIMAL, ALMOST_LOCALLY_SOLVED) && return :inexact
    st in (INFEASIBLE, LOCALLY_INFEASIBLE) && return :infeasible_reported
    st == ALMOST_INFEASIBLE && return :inexact_infeasible
    return :failed
end

# Emptiness of the polyhedron {Ain p ≤ bin, E p = e} (E_r ∩ Γ), decided by a separate LP solved with
# Clarabel: `:infeasible` only if Clarabel returns a primal infeasibility certificate, `:nonempty`
# if it finds a feasible point, `:undecided` otherwise. The certificate is numerical.
function certify_empty(Ain, bin, E, e)
    nθ = size(Ain, 2)
    model = Model(Clarabel.Optimizer)
    set_silent(model)
    @variable(model, θ[1:nθ])
    @constraint(model, Ain * θ .<= bin)
    isempty(e) || @constraint(model, E * θ .== e)
    optimize!(model)
    st = termination_status(model)
    st == INFEASIBLE && return :infeasible
    st in (OPTIMAL, ALMOST_OPTIMAL) && return :nonempty
    return :undecided
end

@doc raw"""
    select_optimal_gne(sol, game, mpavi, ϕ, is_quadratic, optimizer=Clarabel.Optimizer;
                       verify=false, assume_convex=false, Bγ=zeros(m,0), Γ=nothing, game_at=γ->game)

Algorithm 1 (optimal GNE selection). For every critical region with a consistent equal-slack
equality solve
```math
\min_{p=(\theta,\gamma)}\ \phi(K_r p + h_r, \gamma) \quad \text{s.t.}\ p \in \mathcal E_r,\ \gamma\in\Gamma
```
(see [`gne_regions`](@ref)), and keep the best regional solution. Without a leader, `ϕ(x)` takes
the joint action only; with a leader (see `select_optimal_gne(sol, lp::LeaderParametrizedGNEP, …)`)
it is called as `ϕ(x, γ)`. Γ is the γ-box of the mpAVI, optionally intersected with the polytope
`Γ = (A = Aγ, b = bγ)`.

**Represented set.** The selection is over the *represented* GNE set
``\bigcup_r \{K_r p + h_r : p \in \mathcal E_r\}``, i.e. the GNEs whose equal-slack allocation lies
in the parameter box of the mpAVI. GNEs whose allocation lies outside the box are not represented
(enlarge `θub`/`θlb`); `box_boundary = true` flags a selected allocation on the box boundary, where
such truncation may have changed the result.

Statuses are kept separate:
- `map_status`: ParametricDAQP status of the explicit map (`:Solved` = full enumeration of the box);
- `region_status[r]`: `:empty_inconsistent` (equality numerically inconsistent), `:infeasible`
  (``\mathcal E_r\cap\Gamma = \varnothing``, certified by a Clarabel primal infeasibility certificate
  of a separate LP), `:solved`, `:inexact` (almost optimal; kept as a candidate),
  `:inexact_infeasible`, `:infeasible_uncertified` (the regional solver reported infeasibility but
  the LP could not certify it) or `:failed` (any other outcome). Only `:infeasible` and
  `:empty_inconsistent` are read as empty sets;
- `convexity`: `:verified` (quadratic ϕ whose reduced Hessian on the null space of every
  regional equality is numerically positive semidefinite), `:assumed` (caller passed
  `assume_convex = true` and no nonconvexity was detected), `:nonconvex` (a reduced Hessian has a
  negative eigenvalue; that region is solved locally with Ipopt instead of Clarabel), or `:unknown`;
- `status`: `:optimal` if the map is complete, every regional problem was solved or certified empty,
  and convexity is `:verified` or `:assumed` — this means optimal over the represented GNE set,
  conditional on complete enumeration and valid regional optimization; it does not establish
  optimality over GNEs outside the parameter box. `:unverified` if the regional solves are clean
  but convexity is `:nonconvex` or `:unknown` (the result is the best regional KKT point; neither
  global nor local optimality over the GNE set is established). `:partial` otherwise. An error is
  raised if no regional problem was solved (with a distinct message if every region was certified
  empty, i.e. no GNE is represented in the parameter box).
A regional solution whose residual in ``\mathcal E_r`` exceeds `1e-6` is kept as a candidate
with region status `:inexact`. With `verify = true` the selected point is certified with
[`is_gne`](@ref); if the numerical certificate fails, the status is downgraded to `:unverified`.
"""
function select_optimal_gne(
    sol::ParametricDAQP.Solution,
    game::StaticGNEP,
    mpavi::mpAVI,
    ϕ::Function,
    is_quadratic::Bool,
    optimizer = Clarabel.Optimizer;
    verify::Bool = false,
    assume_convex::Bool = false,
    Bγ::AbstractMatrix = zeros(length(game.b_sh), 0),
    Γ = nothing,
    game_at::Function = γ -> game,
    leader_objective::Bool = false
)::OptimalGNEResult
    if !is_quadratic && optimizer == Clarabel.Optimizer
        optimizer = Ipopt.Optimizer
    end
    nγ = size(Bγ, 2)
    ϕxγ = leader_objective ? ϕ : (x, γ) -> ϕ(x)
    candidates = []
    region_status = Dict{Int,Symbol}()
    region_convexity = Symbol[]
    for R in gne_regions(sol, game, mpavi; Bγ=Bγ, include_inconsistent=true)
        if !R.consistent
            region_status[R.region] = :empty_inconsistent
            continue
        end
        n_θ = size(R.K, 2)
        n_u = size(R.K, 1)
        γof(p) = p[n_θ-nγ+1:n_θ]
        Ain, bin = R.Ain, R.bin
        if !isnothing(Γ)
            Ain = [Ain; [zeros(size(Γ.A, 1), n_θ - nγ) Γ.A]]
            bin = [bin; Γ.b]
        end
        # convexity of the regional objective on E_r (quadratic ϕ only), decided before choosing the backend
        conv, backend, P, q = :unknown, optimizer, nothing, nothing
        if is_quadratic
            probe = Model()
            @variable(probe, θp[1:n_θ])
            P, q = quadratic_coefficients(ϕxγ(R.K * θp .+ R.h, θp[n_θ-nγ+1:n_θ]), θp)
            Z = isempty(R.e) ? Matrix(1.0I, n_θ, n_θ) : nullspace(R.E)
            conv = (size(Z, 2) == 0 || minimum(eigvals(Symmetric(Z' * P * Z))) >= -1e-9) ? :verified : :nonconvex
            # Clarabel needs a convex objective in the full parameter space: otherwise solve locally with Ipopt
            if backend == Clarabel.Optimizer && minimum(eigvals(Symmetric(P))) < -1e-9
                backend = Ipopt.Optimizer
            end
        end
        model = Model(backend)
        set_silent(model)
        if backend == Clarabel.Optimizer
            # the certified sets are typically lower dimensional: solve to high accuracy
            for att in ("tol_gap_abs", "tol_gap_rel", "tol_feas")
                set_attribute(model, att, 1e-10)
            end
        end
        @variable(model, θ[1:n_θ])
        @constraint(model, Ain * θ .<= bin)
        isempty(R.e) || @constraint(model, R.E * θ .== R.e)
        if is_quadratic
            @objective(model, Min, ϕxγ(R.K * θ .+ R.h, θ[n_θ-nγ+1:n_θ]))
        else
            @variable(model, y[1:n_u + nγ])
            @constraint(model, y[1:n_u] .== R.K * θ .+ R.h)
            @constraint(model, y[n_u+1:end] .== θ[n_θ-nγ+1:n_θ])
            @operator(model, op_ϕ, n_u + nγ, (ys...) -> ϕxγ(collect(ys[1:n_u]), collect(ys[n_u+1:end])))
            @objective(model, Min, op_ϕ(y...))
        end
        optimize!(model)
        st = regional_status(model)
        if st == :infeasible_reported
            # a (local) infeasibility report is not a certificate: decide emptiness by a separate LP
            ce = certify_empty(Ain, bin, R.E, R.e)
            st = ce == :infeasible ? :infeasible : :infeasible_uncertified
        end
        region_status[R.region] = st
        st in (:solved, :inexact) || continue
        push!(region_convexity, conv)
        ϕp(p) = ϕxγ(R.K * p + R.h, γof(p))
        θ_opt = is_quadratic ? polish_regional_solution(value.(θ), R, mpavi, ϕp, Ain, bin; P=P, q=q) :
                               polish_regional_solution(value.(θ), R, mpavi, ϕp, Ain, bin)
        # the polished point may have been rejected: check the candidate against E_r ∩ Γ itself
        resid = max(maximum(Ain * θ_opt - bin; init=0.0), isempty(R.e) ? 0.0 : norm(R.E * θ_opt - R.e, Inf))
        if resid > 1e-6
            st = :inexact
            region_status[R.region] = st
        end
        u_opt = R.K * θ_opt + R.h
        push!(candidates, (θ=θ_opt, u=u_opt, γ=γof(θ_opt), φ=ϕp(θ_opt), region=R.region, status=st, convexity=conv))
    end
    if isempty(candidates)
        all(st -> st in (:infeasible, :empty_inconsistent), values(region_status)) &&
            error("[select_optimal_gne] no GNE is represented: every certified set is empty (no equal-slack allocation of a GNE lies in the parameter box, intersected with Γ if given)")
        error("[select_optimal_gne] no regional problem was solved; region statuses: $(region_status)")
    end
    best = candidates[argmin([c.φ for c in candidates])]
    map_status = sol.status
    convexity = any(==(:nonconvex), region_convexity) ? :nonconvex :
                (is_quadratic && all(==(:verified), region_convexity)) ? :verified :
                assume_convex ? :assumed : :unknown
    numerically_clean = all(st -> st in (:solved, :infeasible, :empty_inconsistent), values(region_status))
    status = !(map_status == :Solved && numerically_clean) ? :partial :
             convexity in (:verified, :assumed) ? :optimal : :unverified
    # boundary diagnostic: an allocation on the parameter box may hide GNEs outside the box
    nβ = length(best.θ) - nγ
    box_boundary = any(k -> min(best.θ[k] - mpavi.lb[k], mpavi.ub[k] - best.θ[k]) <= 1e-7, 1:nβ)
    box_boundary && @warn "[select_optimal_gne] the selected allocation lies on the boundary of the parameter box: GNEs whose equal-slack allocation lies outside the box are not represented; enlarge θub/θlb to check" best.θ[1:nβ]
    if verify
        cert = is_gne(game_at(best.γ), best.u)
        if !cert.ok
            @warn "[select_optimal_gne] selected point failed the numerical GNE certificate; status set to :unverified" cert.gap cert.infeas
            status == :optimal && (status = :unverified)
        end
    end
    return OptimalGNEResult(best.θ, best.u, best.φ, best.region, candidates, best.γ, status, map_status,
                            region_status, convexity, box_boundary)
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