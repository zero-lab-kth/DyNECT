####### Leader-parametrized static GNEP: parameter p = [θ; γ] #######

"""
    fix_leader(lp::LeaderParametrizedGNEP, γ) -> StaticGNEP

The GNEP at leader value `γ`: `q_i ← q_i + S_i γ`, `b_sh ← b_sh + Bγ γ`.
"""
function fix_leader(lp::LeaderParametrizedGNEP, γ::AbstractVector)
    g = lp.gnep
    offs = cumsum([0; g.n])
    q = [g.q[i] .+ lp.S[offs[i]+1:offs[i+1], :] * γ for i in 1:g.N]
    return StaticGNEP(g.Q, q, g.A_loc, g.b_loc, g.A_sh, g.b_sh .+ lp.Bγ * γ)
end

"""
    NabetaniParametrization(lp::LeaderParametrizedGNEP; θub, θlb, γub, γlb) -> mpAVI

Nabetani mpAVI on `p = [θ; γ]`: `VI(Hx + f + [0 S] p, A x ≤ [B Tγ] p + b)`, with `Bγ` on player N's
shared rows.
"""
function NabetaniParametrization(lp::LeaderParametrizedGNEP;
    θub::Union{Vector{Float64},Nothing}=nothing, θlb::Union{Vector{Float64},Nothing}=nothing,
    γub::Vector{Float64}, γlb::Vector{Float64})
    @assert length(γub) == length(γlb) == lp.n_γ "γ bounds must have length n_γ = $(lp.n_γ)"
    base = NabetaniParametrization(lp.gnep; θub=θub, θlb=θlb)
    g = lp.gnep
    m_sh = length(g.b_sh)
    rows_N = sum(size(A, 1) for A in g.A_loc) + (g.N - 1) * m_sh .+ (1:m_sh)
    Tγ = zeros(base.m, lp.n_γ)
    Tγ[rows_N, :] = lp.Bγ
    return mpAVI(Matrix{Float64}(base.H), [base.F lp.S], base.f, base.A, [base.B Tγ], base.b;
        C=zeros(0, base.n_θ + lp.n_γ), d=zeros(0), ub=[base.ub; γub], lb=[base.lb; γlb])
end

# StaticGNEP methods with the γ data taken from the leader game
gne_regions(sol::ParametricDAQP.Solution, lp::LeaderParametrizedGNEP, mpavi::mpAVI; include_inconsistent::Bool=false) =
    gne_regions(sol, lp.gnep, mpavi; Bγ=lp.Bγ, include_inconsistent=include_inconsistent)

equal_slack_allocation(lp::LeaderParametrizedGNEP, x::AbstractVector, γ::AbstractVector) =
    equal_slack_allocation(lp.gnep, x; γ=γ, Bγ=lp.Bγ)

is_gne(lp::LeaderParametrizedGNEP, x::AbstractVector, γ::AbstractVector; tol::Float64=1e-7) =
    is_gne(fix_leader(lp, γ), x; tol=tol)

in_gne_map(regions::AbstractVector, lp::LeaderParametrizedGNEP, x::AbstractVector, γ::AbstractVector; tol::Float64=1e-8) =
    in_gne_map(regions, lp.gnep, x; γ=γ, Bγ=lp.Bγ, tol=tol)

check_gne_map(sol::ParametricDAQP.Solution, lp::LeaderParametrizedGNEP, mpavi::mpAVI; tol::Float64=1e-7) =
    check_gne_map(sol, lp.gnep, mpavi; Bγ=lp.Bγ, game_at=γ -> fix_leader(lp, γ), tol=tol)

"""
    select_optimal_gne(sol, lp::LeaderParametrizedGNEP, mpavi, ϕ, is_quadratic, optimizer=Clarabel.Optimizer;
                       Γ=nothing, verify=false, assume_convex=false)

Algorithm 1 with a leader: minimizes `ϕ(x, γ)` over the GNEs and `γ` (γ-box ∩ `Γ = (A, b)`).
"""
select_optimal_gne(sol::ParametricDAQP.Solution, lp::LeaderParametrizedGNEP, mpavi::mpAVI, ϕ::Function,
                   is_quadratic::Bool, optimizer=Clarabel.Optimizer; Γ=nothing, verify::Bool=false,
                   assume_convex::Bool=false) =
    select_optimal_gne(sol, lp.gnep, mpavi, ϕ, is_quadratic, optimizer; verify=verify, assume_convex=assume_convex,
                       Bγ=lp.Bγ, Γ=Γ, game_at=γ -> fix_leader(lp, γ), leader_objective=true)
