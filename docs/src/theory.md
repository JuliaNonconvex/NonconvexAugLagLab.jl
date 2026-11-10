# Theory: augmented Lagrangians, PBM and quasi-Newton sub-solvers

## The augmented Lagrangian method

For the equality-constrained problem $\min f(x)$ s.t. $c(x) = 0$, the
**augmented Lagrangian** (Hestenes; Powell; Rockafellar) is

$$L_\rho(x, y) = f(x) + y^\top c(x) + \tfrac{\rho}{2}\lVert c(x)\rVert^2.$$

The outer loop alternates:

1. **Primal step**: $x^{k+1} \approx \arg\min_x L_{\rho_k}(x, y^k)$.
2. **Multiplier update**: $y^{k+1} = y^k + \rho_k\, c(x^{k+1})$.

The update rule is forced by stationarity: the KKT system of the original
problem pairs $\nabla f + J^\top y = 0$ with $c = 0$, and the subproblem
stationarity $\nabla f(x^{k+1}) + J^\top (y^k + \rho_k c(x^{k+1})) = 0$
shows the parenthesized quantity *is* the next multiplier estimate.
Rockafellar's interpretation: the ALM is the **proximal point method on the
dual function**, which is why it converges under far weaker assumptions
than the pure penalty method — the penalty $\rho$ no longer needs to tend
to infinity, and a constant, large-enough $\rho$ suffices for local
convergence. Inequality constraints enter through the max-form
$g(x) \le 0 \leadsto \lambda^\top \max(g,0) + \frac{\rho}{2}\lVert\max(g,0)\rVert^2$
with the update $\lambda^{k+1} = \max(0, \lambda^k + \rho_k g(x^{k+1}))$.

## The practical ALM of Birgin–Martínez

[`AugLagAlg`](@ref) follows the practical scheme of Birgin & Martínez
(ALGENCAN):

- **Feasibility-driven penalty.** The penalty is increased *only when the
  feasibility did not improve enough*. Two tolerance sequences drive the
  loop: $\eta_k$ (feasibility target — a step is "good" when
  $\lVert v(x)\rVert_\infty \le \eta_k$) and $\omega_k$ (how accurately the
  subproblem must be solved). After a good step both tighten with $1/\rho$;
  after a bad step the penalty grows geometrically and the tolerances
  reset.
- **Multiplier safeguards.** The multipliers are updated only on good
  steps, avoiding the classical failure where noisy multipliers from
  poorly solved subproblems destabilize the next one.
- **Two-way adaptation (Curtis–Gould–Jiang–Robinson).** A monotonically
  increasing penalty makes the subproblems ill-conditioned. When the
  multiplier update is small and feasibility improved, the penalty is
  allowed to *decrease*, keeping the subproblems tractable.
- **Infeasibility detection.** If the penalty reaches its cap while the
  violation stagnates, the algorithm stops rather than looping forever.
- **Least-squares warm start** (`lsq_multipliers`): the first multipliers
  solve $\min_\lambda \lVert \nabla f(x^0) + J(x^0)^\top \lambda \rVert^2$,
  a far better estimate than zero for equality-constrained problems.

## The penalty-barrier-multiplier method (PBM)

The classical ALM treats violations quadratically. The PBM of Ben-Tal &
Zibulevsky (used by PENNON) replaces the inequality term with a shifted
logarithmic barrier:

$$F(x, u, c) = f(x) + \sum_i c_i \log\!\left(1 + \frac{u_i g_i(x)}{c_i}\right), \qquad
\frac{u_i g_i(x)}{c_i} > -1,$$

with the multiplier update $u_i^{k+1} = u_i^k \,/\, (1 + u_i^k g_i(x^k)/c_i)$
and a penalty increase when progress stalls. Stationarity of the
subproblem gives $\nabla f + \sum_i \frac{u_i}{1 + u_i g_i / c_i} \nabla g_i = 0$,
so the effective multiplier is $\lambda_i = u_i/(1 + u_i g_i/c_i)$ — the
fixed point of the update is exactly the KKT complementarity, and on the
feasible side ($g_i < 0$) the multipliers *grow*, sharpening the barrier
near the boundary. In [`AugLagAlg`](@ref) the PBM term applies on the
feasible side while the infeasible side falls back to the classical
quadratic term — the standard globalization of the barrier domain
(`penalty = :pbm`, or `:mixed` for quadratic semidefinite terms).

## Projected ALM for semidefinite constraints

With a cone constraint $P(x) \succeq 0$ and a matrix multiplier
$Y \succeq 0$, the projected augmented Lagrangian (Sun–Sun–Zhang) is

$$\varphi(x; Y, \rho) = \frac{1}{2\rho}\Big(\lVert \Pi_{\mathbb{S}_+^n}(Y - \rho P(x))\rVert_F^2 - \lVert Y\rVert_F^2\Big),$$

with the multiplier update $Y^{k+1} = \Pi_{\mathbb{S}_+^n}(Y^k - \rho_k P(x^{k+1}))$.
The squared-distance-to-cone penalty is C¹ with gradient
$-\langle \Pi_{\mathbb{S}_+^n}(Y - \rho P(x)), \nabla P(x)\rangle$, which
NonconvexSemidefinite's `project_psd` rrule differentiates exactly. The
scheme inherits the local (super)linear convergence rates of the classical
ALM, and the subproblems are smooth NLPs solvable by any of the sub-solvers
— including the built-in quasi-Newton driver.

## Quasi-Newton sub-solvers

When every constraint is relaxed, each ALM subproblem is a smooth
box-constrained NLP, solved by the built-in [`QuasiNewtonAlg`](@ref). The
theory:

- **Secant equation.** A quasi-Newton model $B$ of the Hessian must satisfy
  $B_{k+1} s_k = y_k$ with $s_k = x^{k+1} - x^k$ and $y_k = \nabla f^{k+1} - \nabla f^k$.
- **BFGS.** The minimal-norm update preserving symmetry and the secant
  equation is
  $B_{k+1} = B_k + \frac{y y^\top}{y^\top s} - \frac{B_k s s^\top B_k}{s^\top B_k s}$,
  with the Sherman–Morrison–Woodbury **inverse** update
  $H_{k+1} = (I - \rho s y^\top) H_k (I - \rho y s^\top) + \rho s s^\top$,
  $\rho = 1/(y^\top s)$ — the default `:inverse` form stores $H_k$, making
  the Newton step a single matrix–vector product; `form = :hessian` stores
  $B_k$ and factorizes.
- **Powell damping.** If $y^\top s \le 0$ (nonconvex curvature), $y$ is
  blended toward $B_k s$ so the updated model stays positive definite.
- **L-BFGS.** The two-loop recursion computes $B_k^{-1} g$ (and $B_k v$)
  from the last $m$ pairs in $O(mn)$ without storing a matrix; the base
  operator $H_0$ is the **preconditioner**
  ([`IdentityPreconditioner`](@ref), [`DiagonalPreconditioner`](@ref), or a
  user-defined operator).
- **Structured/sparse BFGS.** The dense BFGS update destroys sparsity. The
  structure-preserving update of Fletcher restricts the dense update to the
  structural nonzeros of the container, so a sparse (or user-defined)
  Hessian keeps its pattern and its factorization works unchanged.
- **Trust region.** The subproblem $\min_{\lVert d\rVert \le \Delta} g^\top d + \frac12 d^\top B d$
  is solved with the Steihaug–Toint conjugate-gradient method, which
  terminates on the boundary or on negative curvature — robust for
  indefinite models. Paired with [`NewtonTRApprox`](@ref) (the exact Hessian
  through NonconvexCore's AD backends) this is the Newton trust-region
  method; the step is projected onto the box and accepted by the
  ratio of actual to predicted reduction.
- **Globalization.** Line searches use Armijo backtracking along the
  projected direction; bounds are respected by projection, with the
  standard caveat that a projected step can stall on the boundary — the
  driver then shrinks the step or the radius.

The whole machinery is user-extensible through the
[`AbstractHessianApprox`](@ref) interface: `init!`, `apply!`, `update!`,
`solve!` and `reset!` — a custom matrix type with custom secant
accumulation and a custom factorization plugs in directly.
