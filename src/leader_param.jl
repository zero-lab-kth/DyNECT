####### Leader-parametrized static GNEP (γ-extension of the Nabetani parametrization) #######
# The leader variable γ enters the pseudo-gradient (S γ) and the shared right-hand side (Bγ γ):
#
#     F(x; γ)  = H x + f + S γ,        Σ_i A_sh,i x_i ≤ b_sh + Bγ γ
#
# The joint parameter of the mpAVI is p = [θ; γ] (θ first, γ last). θ = col(β_1, …, β_{N-1})
# are the allocations of agents 1, …, N-1; agent N receives β_N = b_sh + Bγ γ − Σ_{i<N} θ_i,
# so Σ_i β_i = b_sh + Bγ γ. The type `LeaderParametrizedGNEP` is defined in types.jl.

"""
    fix_leader(lp::LeaderParametrizedGNEP, γ) -> StaticGNEP

The ordinary GNEP obtained by fixing the leader variable: `q_i ← q_i + S_i γ`,
`b_sh ← b_sh + Bγ γ`.
"""
function fix_leader(lp::LeaderParametrizedGNEP, γ::AbstractVector)
    g = lp.gnep
    @assert length(γ) == lp.n_γ "γ must have length n_γ = $(lp.n_γ), got $(length(γ))"
    offsets = cumsum(vcat(0, g.n))
    q = [g.q[i] .+ lp.S[offsets[i]+1:offsets[i+1], :] * γ for i in 1:g.N]
    return StaticGNEP(g.Q, q, g.A_loc, g.b_loc, g.A_sh, g.b_sh .+ lp.Bγ * γ)
end

"""
    NabetaniParametrization(lp::LeaderParametrizedGNEP; θub, θlb, γub, γlb) -> mpAVI

Nabetani parametrization of the γ-family over the joint parameter `p = [θ; γ]`:
`VI(Hx + f + [0 S] p, A x ≤ [B_g Tγ] p + b)`, where `Tγ` places `Bγ` on player N's
shared-constraint rows. The θ part is exactly upstream's `NabetaniParametrization(lp.gnep)`.
"""
function NabetaniParametrization(lp::LeaderParametrizedGNEP;
    θub::Union{Vector{Float64},Nothing}=nothing, θlb::Union{Vector{Float64},Nothing}=nothing,
    γub::Vector{Float64}, γlb::Vector{Float64})

    @assert length(γub) == lp.n_γ && length(γlb) == lp.n_γ "γ bounds must have length n_γ = $(lp.n_γ)"
    base = NabetaniParametrization(lp.gnep; θub=θub, θlb=θlb)

    # Player N's copy of the shared constraints: rows after the local rows and the
    # first N-1 players' copies (same layout as in NabetaniParametrization)
    g = lp.gnep
    m_sh = length(g.b_sh)
    n_local = sum(size(g.A_loc[i], 1) for i in 1:g.N)
    rows_N = n_local + (g.N - 1) * m_sh .+ (1:m_sh)
    Tγ = zeros(base.m, lp.n_γ)
    Tγ[rows_N, :] = lp.Bγ

    return mpAVI(Matrix{Float64}(base.H), [base.F lp.S], base.f, base.A, [base.B Tγ], base.b;
        C=zeros(0, base.n_θ + lp.n_γ), d=zeros(0),
        ub=[base.ub; γub], lb=[base.lb; γlb])
end



####### Equal-slack certification and selection for the leader-parametrized game #######
# Thin wrappers: the γ data (Bγ, S) are read from the leader game itself, so no state is
# duplicated in the solver. Parameters are p = [θ; γ].

"""
    gne_regions(sol, lp::LeaderParametrizedGNEP, mpavi; include_inconsistent=false)

Equal-slack certified sets ``\\mathcal E_r`` on ``p = (θ, γ)``; see the `StaticGNEP` method.
"""
gne_regions(sol::ParametricDAQP.Solution, lp::LeaderParametrizedGNEP, mpavi::mpAVI; include_inconsistent::Bool=false) =
    gne_regions(sol, lp.gnep, mpavi; Bγ=lp.Bγ, include_inconsistent=include_inconsistent)

"""
    equal_slack_allocation(lp::LeaderParametrizedGNEP, x, γ) -> θ

Free allocations ``θ = β_{1:N-1}`` of the equal-slack allocation at leader value `γ`.
"""
equal_slack_allocation(lp::LeaderParametrizedGNEP, x::AbstractVector, γ::AbstractVector) =
    equal_slack_allocation(lp.gnep, x; γ=γ, Bγ=lp.Bγ)

"""
    is_gne(lp::LeaderParametrizedGNEP, x, γ; tol=1e-7)

GNE certificate of `x` for the game at leader value `γ` (`is_gne(fix_leader(lp, γ), x)`).
"""
is_gne(lp::LeaderParametrizedGNEP, x::AbstractVector, γ::AbstractVector; tol::Float64=1e-7) =
    is_gne(fix_leader(lp, γ), x; tol=tol)

"""
    in_gne_map(regions, lp::LeaderParametrizedGNEP, x, γ; tol=1e-8)

Numerical completeness check at leader value `γ`; see the `StaticGNEP` method.
"""
in_gne_map(regions::AbstractVector, lp::LeaderParametrizedGNEP, x::AbstractVector, γ::AbstractVector; tol::Float64=1e-8) =
    in_gne_map(regions, lp.gnep, x; γ=γ, Bγ=lp.Bγ, tol=tol)

"""
    check_gne_map(sol, lp::LeaderParametrizedGNEP, mpavi; tol=1e-7)

Numerical soundness check; every sampled point is certified on `fix_leader(lp, γ)`.
"""
check_gne_map(sol::ParametricDAQP.Solution, lp::LeaderParametrizedGNEP, mpavi::mpAVI; tol::Float64=1e-7) =
    check_gne_map(sol, lp.gnep, mpavi; Bγ=lp.Bγ, game_at=γ -> fix_leader(lp, γ), tol=tol)

"""
    select_optimal_gne(sol, lp::LeaderParametrizedGNEP, mpavi, ϕ, is_quadratic, optimizer=Clarabel.Optimizer;
                       Γ=nothing, verify=false, assume_convex=false)

Algorithm 1 with a leader: minimizes `ϕ(x, γ)` jointly over the GNEs and `γ ∈ Γ` (γ-box of the
mpAVI, optionally intersected with `Γ = (A = Aγ, b = bγ)`). Returns the optimal joint action in
`u_star` and the leader value in `γ_star`; statuses as in the `StaticGNEP` method.
"""
select_optimal_gne(sol::ParametricDAQP.Solution, lp::LeaderParametrizedGNEP, mpavi::mpAVI, ϕ::Function,
                   is_quadratic::Bool, optimizer=Clarabel.Optimizer; Γ=nothing, verify::Bool=false,
                   assume_convex::Bool=false) =
    select_optimal_gne(sol, lp.gnep, mpavi, ϕ, is_quadratic, optimizer; verify=verify, assume_convex=assume_convex,
                       Bγ=lp.Bγ, Γ=Γ, game_at=γ -> fix_leader(lp, γ), leader_objective=true)
