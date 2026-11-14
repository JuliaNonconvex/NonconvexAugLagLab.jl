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

## The hardening stack (defaults in `AugLagAlg`)

On top of the Birgin–Martínez loop, [`AugLagAlg`](@ref) enables a set of
hardening devices, each benchmark-gated on the TopOpt local-stress and
Rosetta AC-OPF problems. They are the defaults because the measurements
favored them; several (marked *opt-in*) are available but off by default
because they trade one benchmark against the other.

### Jacobian-free constraint scaling (`scaling = true`)

Before the ALM starts, each relaxed constraint is normalized by the
magnitude of its gradient at the initial point,
$\tilde c_i(x) = c_i(x)/\lVert\nabla c_i(x^0)\rVert$, and the objective by its
initial magnitude. The penalty then acts on *balanced, unit-scaled*
violations, so a single $\rho$ makes sense for all constraints — without the
scaling, constraint groups with wildly different gradient scales need
wildly different penalties, which is the classic source of ALM divergence
on problems like AC-OPF (Curtis–Jiang–Robinson). The scaling uses only
$Jv$ products (finite-difference), never a materialized Jacobian.

### Matrix-free CG least-squares multiplier initialization (`lsq_multipliers = true`)

The first equality multipliers solve
$\min_\lambda \lVert \nabla f(x^0) + J(x^0)^\top \lambda\rVert^2$ — the
stationarity residual in multiplier space — by CGLS on the normal
equations, again using only $Jv$ and $J^\top v$ products. The cost is a
handful of constraint evaluations and the payoff is a multiplier estimate
orders of magnitude better than zero, which removes the first many
expensive outer iterations.

### Safeguarded multipliers and selective penalty weights

The multiplier update is clamped ($|y|, \lambda \le 10^8$) so a poorly
solved subproblem cannot poison the next one with a huge multiplier.
Instead of escalating the *global* penalty when a single constraint group
is violated, the **selective** update (`weights`) bumps only the weights of
the most-violated groups (by `ρ_incr`, capped at `ρ_max/ρ`), escalating
$\rho$ itself only when no single group stands out. This keeps the
well-behaved groups' subproblems well-conditioned while the offender is
penalized harder — the per-group generalization of the Birgin–Martínez
feasibility-driven update.

### Fletcher–Leyffer filter

The outer loop is globalized by a **filter**, not a merit function: a
candidate step is rejected only if a previous iterate dominates it in
*both* the AL merit and the constraint violation. This avoids the classic
failure where a single scalar merit value misweights feasibility against
optimality and rejects perfectly good steps (Fletcher–Leyffer 2002). The
filter is reset when the penalty changes scale.

### Anderson acceleration on the multiplier map (depth 2)

The multiplier update $y^{k+1} = y^k + \rho\, c(x^{k+1})$ is a fixed-point
iteration, and near a solution its Jacobian is contractive (its spectral
radius shrinks like $O(1/\rho)$). Anderson acceleration mixes the last $m$
fixed-point iterates, $y^{k+1} = \sum_j \theta_j (y^j + R^j)$ with the
least-squares weights $\theta$ minimizing $\lVert\sum_j \theta_j R^j\rVert$,
$R^j = (y^{j+1} - y^j)$ — an $m \times m$ solve per outer iteration that
typically halves the outer count. The depth (2) and weight clamping
($\theta \in [-2, 2]$) are conservative defaults.

### Penalty-scaled, warm-started L-BFGS (`memory = 50`)

The built-in sub-solver keeps *one* L-BFGS history across the whole outer
loop (`reset = false`): the sequence of subproblems is a homotopy in
$\rho$, so the secant pairs of the previous subproblem are excellent
curvature for the next one. The two-loop base operator is the
**penalty-scaled diagonal** $H_0 = (1+\rho)^{-1} I$: the AL objective's
curvature grows like $\rho$ on the constraint subspace, and an $H_0$ that
shrinks with $1/(1+\rho)$ keeps the first CG steps of each subproblem on
the right scale. The memory is a deliberate default of 50. Measured on the
released 0.2.5 code: a small memory (5) is comparable but never better; a
larger memory (100) leaves the TopOpt stress design essentially unchanged
(+0.7% objective, +47% time) but materially improves equality-heavy
AC-OPF — on PGLib case5 the gap drops 0.12 → 0.067 and the max scaled
violation 1.14 → 0.70 at ~5× the wall-clock. Use `memory = 100` when
constraint feasibility dominates runtime.

### Levenberg damping and stall retry

Inside the Steihaug–Toint solve the curvature is damped $B + \tau I$ with
$\tau$ adapted by the trust-region ratio (Levenberg regularization): when
steps are rejected, $\tau$ grows and the model becomes more conservative;
when they succeed, $\tau$ shrinks back toward the secant curvature. A
**stall retry** detects a subproblem that stopped progressing (the
projected gradient no longer decreases) and restarts it with a relaxed
tolerance rather than burning the outer budget on a dead solve.

### TRON inner solver (`linesearch = :tron`, *opt-in*)

As an alternative to the Steihaug–Toint trust-region step, the subproblem
$\min_{l \le x \le u,\; \lVert d\rVert \le \Delta} g^\top d + \frac12 d^\top B d$
can be solved by **TRON** (Lin & Moré 1999): projected conjugate gradients
on the free-variable subspace with active-set identification — a variable
that hits a bound mid-sweep is fixed for the rest of the sweep, negative
curvature rides to the trust-region boundary, and the step respects both
the bounds and the radius at every CG iterate. Measured on the released
0.2.5 code (after the L-BFGS matvec fix): on the TopOpt local-stress quick
config it is 27% slower with a 1.8% worse objective; on PGLib case5 it
improves the gap (0.12 → 0.095) and the max scaled violation (1.14 → 0.95)
at comparable time, and on case14 it slightly improves feasibility
(0.0144 → 0.013). It stays opt-in for problems where constraint
feasibility dominates TopOpt-style speed.

### Elastic mode (`elastic = true`, *opt-in*)

For equality-constrained problems whose subproblems risk infeasibility,
the equalities are relaxed through slack variables,
$y^\top(c - s) + \frac{\rho}{2}\lVert c - s\rVert^2 + \mu\sum_i\sqrt{s_i^2 + \epsilon^2}$,
the elastic ALM: $s$ is solved along with $x$ in every subproblem, and the
slack penalty $\mu$ absorbs constraint infeasibility. Off by default
because the measured benchmarks do not need it.

### Warm continuation (`warm_continue = true`, *opt-in*)

`optimize!` can be called repeatedly on the same `Workspace`, reusing the
previous run's final iterate, multipliers (rescaled to the new constraint
scales), per-group weights and the L-BFGS curvature object — exactly the
state a continuation scheme needs. $\rho$, $\eta$, $\omega$, the filter and
the Anderson history re-initialize from the new options, and a structure
mismatch falls back to a cold start. Measured on the released 0.2.5 code (TopOpt
local-stress SIMP continuation): warm continuation is consistently faster
than cold restarts (26% on a coarse 3-step schedule, 10% on a fine 6-step
one) and keeps the design at the stress threshold (violation 0.0 vs 0.005
on 3 steps, 0.003 vs 0.032 on 6 steps), while cold restarts end with a
lighter objective but slightly violate the stress limit.

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
