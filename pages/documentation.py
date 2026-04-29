"""
Documentation page for the Ramsey–Cass–Koopmans Model Explorer.

This page explains the economic model, the mathematics behind it,
and how the code implements each component.
"""

import numpy as np
import streamlit as st

st.set_page_config(layout="wide")
st.title("Documentation")
st.caption("How the Ramsey–Cass–Koopmans Model Explorer works — mathematics and code.")

# ─────────────────────────────────────────────────────────────────────────────
tab_econ, tab_math, tab_code = st.tabs([
    "1 · Economic model",
    "2 · Mathematics",
    "3 · Code walkthrough",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ECONOMIC MODEL
# ═══════════════════════════════════════════════════════════════════════════════
with tab_econ:
    st.header("The Economic Model")

    st.markdown("""
    The **Ramsey–Cass–Koopmans (RCK) model** is the canonical model of optimal
    economic growth. It asks: *how should a society split its output between
    consumption today and investment (i.e. saving) for tomorrow?*

    Unlike the simpler Solow model, where the saving rate is fixed exogenously,
    in the RCK model a representative household **optimises** — it chooses a
    consumption path that maximises its total lifetime utility.
    """)

    st.subheader("The actors")
    st.markdown("""
    - **Household** — lives forever, derives utility from consuming $c(t)$ at each
      instant, and discounts the future at rate $\\rho > 0$ (the *time-preference rate*
      or *impatience rate*).
    - **Firm** — hires capital from the household and produces output using a
      Cobb-Douglas technology $y = k^\\alpha$.
    - **Market** — competitive; the rental rate of capital equals its marginal product.
    """)

    st.subheader("Parameters")
    st.markdown("""
    | Symbol | Name | Typical value | Role |
    |--------|------|--------------|------|
    | $\\alpha$ | Capital share | 0.30 | Curvature of production function |
    | $\\rho$ | Time preference | 0.05 | How impatient the household is |
    | $\\delta$ | Depreciation | 0.00–0.10 | Capital wears out at rate $\\delta$ |
    | $k(0)$ | Initial capital | — | Inherited from history, **not a choice** |
    | $c(0)$ | Initial consumption | — | The household's **free variable** |
    """)

    st.subheader("The household's problem")
    st.markdown(
        r"The household maximises discounted log utility (log utility implies "
        r"a unit elasticity of intertemporal substitution):"
    )
    st.latex(r"\max_{c(t) \geq 0} \int_0^\infty e^{-\rho t} \ln c(t)\, dt")
    st.markdown("subject to the **capital accumulation equation**:")
    st.latex(r"\dot{k}(t) = k(t)^\alpha - \delta\, k(t) - c(t)")
    st.markdown(r"""
    Interpretation: whatever you produce ($k^\alpha$) minus what depreciates
    ($\delta k$) minus what you eat ($c$) gets added to the capital stock.
    """)

    st.subheader("The steady state — modified golden rule")
    st.markdown(r"""
    In the **long-run steady state** $(\tilde k, \tilde c)$ both $\dot k = 0$ and
    $\dot c = 0$. The key result is the **modified golden rule**:

    > Capital is accumulated until its *net marginal product* equals the
    > household's impatience:
    > $$\alpha \tilde k^{\alpha-1} - \delta = \rho$$

    This differs from the *golden rule* (which maximises consumption, not utility)
    because impatient households accept a lower capital stock in exchange for
    higher consumption today.
    """)

    st.subheader("Saddle-point dynamics and the transversality condition")
    st.markdown(r"""
    The steady state is a **saddle point**: there is a one-dimensional *stable
    manifold* (the **saddle path**) that leads into it, and a one-dimensional
    *unstable manifold* that leads away.

    The **transversality condition** (no Ponzi-scheme borrowing, no capital
    wastage at infinity) forces the economy onto the saddle path. This pins down
    the initial consumption $c(0)$ uniquely once $k(0)$ is given.

    - **Too high $c(0)$** → capital explodes (over-saving, the household gives
      away too little) — violates transversality.
    - **Too low $c(0)$** → capital collapses to zero (the economy eats its entire
      capital stock).
    - **Exactly right** → the economy converges to $(\tilde k, \tilde c)$ along
      the saddle path.

    The app visualises this knife-edge property interactively.
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MATHEMATICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_math:
    st.header("Mathematics")

    # ── 2.1 The ODE system ───────────────────────────────────────────────────
    st.subheader("2.1  The ODE system")
    st.markdown("The model reduces to two autonomous first-order ODEs in $(k, c)$:")
    st.latex(r"""
    \begin{cases}
    \dot{k} = k^\alpha - \delta k - c \\[6pt]
    \dot{c} = c\bigl(\alpha k^{\alpha-1} - \delta - \rho\bigr)
    \end{cases}
    """)
    st.markdown(r"""
    The **Euler equation** for $\dot c$ is derived from the household's
    first-order condition (Pontryagin maximum principle):

    $$\frac{\dot c}{c} = \underbrace{\alpha k^{\alpha-1}}_{\text{MPK}} - \delta - \rho$$

    When $\text{MPK} > \delta + \rho$ the return on waiting exceeds impatience →
    consumption grows. When $\text{MPK} < \delta + \rho$ → consumption falls.
    """)

    # ── 2.2 Nullclines ──────────────────────────────────────────────────────
    st.subheader("2.2  Nullclines")
    st.markdown(r"""
    Setting each derivative to zero defines two **nullclines**:

    | Nullcline | Equation | Shape |
    |-----------|----------|-------|
    | $\dot k = 0$ | $c = k^\alpha - \delta k$ | Hump-shaped curve peaking at the golden-rule capital $k_{\text{gr}} = (\alpha/\delta)^{1/(1-\alpha)}$ |
    | $\dot c = 0$ | $k = \tilde k$ | Vertical line |

    Their intersection is the steady state. The phase diagram's four quadrants
    have predictable arrow directions determined by the signs of $\dot k$ and $\dot c$.
    """)

    # ── 2.3 Steady state ────────────────────────────────────────────────────
    st.subheader("2.3  Steady state")
    st.markdown(r"Setting $\dot c = 0$ and $\dot k = 0$ gives:")
    st.latex(r"""
    \tilde k = \left(\frac{\alpha}{\rho + \delta}\right)^{\!\frac{1}{1-\alpha}},
    \qquad
    \tilde c = \tilde k^\alpha - \delta\, \tilde k
    """)

    # ── 2.4 Linearisation and Jacobian ──────────────────────────────────────
    st.subheader("2.4  Linearisation and the Jacobian")
    st.markdown(r"""
    Let $\hat k = k - \tilde k$ and $\hat c = c - \tilde c$ be small deviations.
    Linearising the ODE system around the steady state:
    """)
    st.latex(r"""
    \begin{pmatrix} \dot{\hat k} \\ \dot{\hat c} \end{pmatrix}
    = J \begin{pmatrix} \hat k \\ \hat c \end{pmatrix},
    \qquad
    J = \begin{pmatrix}
    \dfrac{\partial \dot k}{\partial k} & \dfrac{\partial \dot k}{\partial c} \\[10pt]
    \dfrac{\partial \dot c}{\partial k} & \dfrac{\partial \dot c}{\partial c}
    \end{pmatrix}_{\!(\tilde k,\,\tilde c)}
    """)
    st.markdown("Computing each partial derivative at the steady state:")
    st.latex(r"""
    J_{11} = \alpha \tilde k^{\alpha-1} - \delta = \rho, \qquad
    J_{12} = -1
    """)
    st.latex(r"""
    J_{21} = \tilde c\,\alpha(\alpha-1)\tilde k^{\alpha-2}, \qquad
    J_{22} = \alpha \tilde k^{\alpha-1} - \delta - \rho = 0
    """)
    st.markdown(r"""
    Note that $J_{11} = \rho$ and $J_{22} = 0$ hold exactly at the steady state
    (by the definition of $\tilde k$). Therefore:
    """)
    st.latex(r"""
    \operatorname{tr}(J) = J_{11} + J_{22} = \rho > 0,
    \qquad
    \det(J) = J_{11} J_{22} - J_{12} J_{21} = J_{21} < 0
    """)
    st.markdown(r"""
    Since $\det(J) < 0$, the Jacobian has **one negative and one positive
    eigenvalue** — confirming the saddle-point structure.
    """)

    # ── 2.5 Eigenvalues ─────────────────────────────────────────────────────
    st.subheader("2.5  Eigenvalues")
    st.latex(r"""
    \lambda_{1,2} = \frac{\operatorname{tr}(J) \pm \sqrt{\operatorname{tr}(J)^2 - 4\det(J)}}{2}
    = \frac{\rho \pm \sqrt{\rho^2 - 4 J_{21}}}{2}
    """)
    st.markdown(r"""
    Because $J_{21} < 0$, the discriminant $\rho^2 - 4 J_{21} > \rho^2 > 0$, so
    both eigenvalues are **real**:

    - $\lambda_1 < 0$ — the **stable** eigenvalue. Perturbations along the
      corresponding eigenvector decay to zero (convergence to steady state).
    - $\lambda_2 > 0$ — the **unstable** eigenvalue. Perturbations along this
      direction grow without bound.

    The **saddle path** is the set of initial conditions $(k(0), c(0))$ whose
    trajectory converges to $(\tilde k, \tilde c)$. Near the steady state it
    is tangent to the stable eigenvector, with slope:
    """)
    st.latex(r"\mu = \lambda_2 = \rho - \lambda_1 > 0")

    # ── 2.6 Bisection ────────────────────────────────────────────────────────
    st.subheader("2.6  Finding c(0) by bisection (shooting method)")
    st.markdown(r"""
    Given a fixed $k(0)$, we need to find the unique $c(0)$ that lies on the
    saddle path. This is a **boundary value problem** — there is no closed-form
    solution for general $k(0)$.

    The **shooting method** converts it to a root-finding problem:

    1. Bracket: pick $[\ell, h]$ with $\ell$ so low the economy collapses
       (trajectory hits $k = 0$ or $c = 0$) and $h$ so high it explodes
       ($k \to \infty$).
    2. **Bisect**: evaluate $\text{mid} = (\ell + h)/2$. Shoot a trajectory
       forward in time.  
       - If it **explodes** → $c(0)$ was too high → new upper bound: $h = \text{mid}$  
       - If it **collapses** → $c(0)$ was too low → new lower bound: $\ell = \text{mid}$  
       - If it **converges** → stop early.
    3. Repeat until the bracket is tight enough.

    Each bisection step **halves the bracket**, so after $n$ steps the error is
    bounded by $(h_0 - \ell_0)/2^n$. With 200 iterations starting from a bracket
    of width $\approx 5$, the precision is better than $10^{-59}$ — far below
    double-precision floating point ($\approx 10^{-16}$).
    """)

    # ── 2.7 Convergence detection ────────────────────────────────────────────
    st.subheader("2.7  Convergence detection")
    st.markdown(r"""
    A trajectory is declared **converged** when it enters a tolerance ball around
    the steady state:
    """)
    st.latex(r"""
    \sqrt{\left(\frac{k - \tilde k}{\tilde k}\right)^2 +
          \left(\frac{c - \tilde c}{\tilde c}\right)^2}
    < \varepsilon_{\text{close}}
    """)
    st.markdown(r"""
    The app uses $\varepsilon_{\text{close}} = 0.01$ for the shooting function
    (1% of steady-state values) and $\varepsilon_{\text{conv}} = 0.02$ for
    displaying the main trajectory.
    """)

    # ── 2.8 Where the Jacobian is actually used ────────────────────────────────
    st.subheader("2.8  Where the Jacobian is used in the app")
    st.markdown(r"""
    The Jacobian and its eigenvalues are computed **only inside the derivations
    panel** (shown when you click *Show/Hide derivations* in the sidebar). They
    are used there to display the numerical values of $J$, $\det J$, $\text{tr}(J)$,
    and $\lambda_{1,2}$ for the current parameter set — purely for educational display.

    The **phase diagram does not draw a saddle-path curve**. The blue trajectory
    you see is the forward-integrated path from $(k(0), c(0))$ as set by the
    sliders. When the bisection animation runs, the final saddle-path trajectory
    is added to the bisection diagram (as a cyan trace) but it is produced by
    `build_paths` with the bisected $c(0)$ — not by backward integration.

    > **Note:** the code contains a `compute_display_saddle()` function that
    > implements backward integration and could draw the exact saddle path, but
    > it is currently not called anywhere in the app.
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CODE WALKTHROUGH
# ═══════════════════════════════════════════════════════════════════════════════
with tab_code:
    st.header("Code Walkthrough")
    st.markdown(
        "All code lives in `ramsey.py`. This page walks through each functional "
        "block in the order it executes."
    )

    # ── 3.1 Constants ────────────────────────────────────────────────────────
    st.subheader("3.1  Global constants")
    st.code("""
T_SHOOT   = 600    # max integration horizon for the shooting function
MAX_STEP  = 0.05   # max ODE step size — controls trajectory smoothness
EXPLODE   = 3.5    # k/k̃ ratio above which a trajectory is declared "exploded"
CLOSE_TOL = 0.01   # distance from SS below which shooting calls "converge"
T_PATH    = 200    # horizon for the main display trajectory
N_EVAL    = 300    # number of points sampled from the dense ODE solution
CONV_TOL  = 0.02   # distance from SS used to mark convergence time t*
""", language="python")
    st.markdown("""
    - `EXPLODE = 3.5` means a trajectory is killed once $k > 3.5\\tilde k$,
      before `solve_ivp` wastes time on clearly divergent paths.
    - `CLOSE_TOL` is deliberately loose (1%) so the integrator stops early
      once the economy is near the steady state; this speeds up the bisection
      significantly.
    """)

    # ── 3.2 ODE factory ─────────────────────────────────────────────────────
    st.subheader("3.2  `make_rhs(alpha, delta, rho)` — ODE factory")
    st.code("""
def make_rhs(alpha_, delta_, rho_):
    def rhs(t, y):
        k, c = y
        dkdt = k**alpha_ - delta_ * k - c
        dcdt = c * (alpha_ * k**(alpha_ - 1.0) - delta_ - rho_)
        return [dkdt, dcdt]
    return rhs
""", language="python")
    st.markdown(r"""
    Returns a closure that captures the parameters. This pattern lets
    `scipy.integrate.solve_ivp` call a single function `rhs(t, y)` without
    passing parameters separately. The two lines implement:

    $$\dot k = k^\alpha - \delta k - c, \qquad \dot c = c(\alpha k^{\alpha-1} - \delta - \rho)$$
    """)

    # ── 3.3 shoot() ─────────────────────────────────────────────────────────
    st.subheader("3.3  `shoot(c0_candidate, k0, k_ss, c_ss, rhs)` — single trajectory")
    st.code("""
def shoot(c0_candidate, k0_, k_ss_, c_ss_, rhs_, t_max=None):
    # Three terminal events stop integration early:
    def ev_explode(t, y):   return y[0] - EXPLODE * k_ss_      # k too large
    def ev_collapse(t, y):  return min(y[0], y[1]) - 1e-10     # k or c → 0
    def ev_converge(t, y):
        k, c = y
        return sqrt(((k-k_ss_)/k_ss_)**2 + ((c-c_ss_)/c_ss_)**2) - CLOSE_TOL

    sol = solve_ivp(rhs_, (0, T_SHOOT), [k0_, c0_candidate],
                    method="RK45", max_step=MAX_STEP,
                    events=[ev_explode, ev_collapse, ev_converge])

    if sol.t_events[2].size > 0: return k_arr, c_arr, "converge"
    if sol.t_events[0].size > 0: return k_arr, c_arr, "explode"
    if sol.t_events[1].size > 0: return k_arr, c_arr, "collapse"
    return k_arr, c_arr, "converge"   # timed-out near SS → also converge
""", language="python")
    st.markdown(r"""
    **Event functions** are the key trick: `solve_ivp` can stop integration the
    moment a user-supplied function crosses zero. This avoids wasting compute
    on clearly divergent trajectories.

    - `ev_explode` fires when $k$ exceeds $3.5\tilde k$.
    - `ev_collapse` fires when either $k$ or $c$ drops below $10^{-10}$.
    - `ev_converge` fires when the Euclidean distance from $(\tilde k, \tilde c)$
      — normalised by steady-state values — drops below `CLOSE_TOL = 0.01`.

    All three are *terminal* (`terminal = True`), so integration stops at the
    first crossing.
    """)

    # ── 3.4 find_c0() ───────────────────────────────────────────────────────
    st.subheader("3.4  `find_c0(alpha, rho, delta, k0, n_iters=200)` — bisection")
    st.code("""
@st.cache_data
def find_c0(alpha_, rho_, delta_, k0_abs_, n_iters=200):
    # Recompute SS for the given parameters
    k_ss_ = (alpha_ / (rho_ + delta_)) ** (1.0 / (1.0 - alpha_))
    c_ss_ = k_ss_**alpha_ - delta_ * k_ss_
    rhs_  = make_rhs(alpha_, delta_, rho_)

    lo, hi = 1e-6, c_ss_ * 2.5   # initial bracket
    for _ in range(n_iters):
        mid = 0.5 * (lo + hi)
        _, _, status = shoot(mid, k0_abs_, k_ss_, c_ss_, rhs_)
        if status == "explode":
            lo = mid      # c(0) was too high
        else:
            hi = mid      # c(0) was too low (collapse) or converged
    return 0.5 * (lo + hi)
""", language="python")
    st.markdown(r"""
    Decorated with `@st.cache_data` — Streamlit memoises the result keyed on all
    arguments, so the expensive 200-iteration bisection only runs once per
    unique combination of $(\alpha, \rho, \delta, k_0)$.

    **Why treat "converge" the same as "collapse" for bracket updates?**
    The saddle path sits at the boundary between explode and everything else.
    Values *above* the saddle path explode; values *at or below* either collapse
    or (if exactly on the path) converge. Narrowing the upper bracket on converge
    is therefore correct.
    """)

    # ── 3.5 build_paths() ───────────────────────────────────────────────────
    st.subheader("3.5  `build_paths(c0, k0, alpha, delta, rho, k_ss, c_ss)` — display trajectory")
    st.code("""
def build_paths(c0_, k0_, alpha_, delta_, rho_, k_ss_, c_ss_):
    sol = solve_ivp(rhs_, (0, T_PATH), [k0_, c0_],
                    method="RK45", max_step=MAX_STEP,
                    events=[ev_converged, ev_explode, ev_collapse],
                    dense_output=True)   # dense_output → interpolatable solution

    # Determine outcome from which event fired first
    ...
    # Sample N_EVAL evenly-spaced points from the dense interpolant
    t_dense = np.linspace(0, t_end, N_EVAL)
    y_dense = sol.sol(t_dense)
    return t_dense, k_path, c_path, t_conv, status
""", language="python")
    st.markdown("""
    `dense_output=True` asks `solve_ivp` to build a continuous interpolant of
    the solution. We then evaluate it at exactly `N_EVAL = 300` evenly-spaced
    points, giving smooth Plotly curves regardless of where the adaptive
    integrator chose its internal steps.
    """)

    # ── 3.6 compute_display_saddle() — dead code note ─────────────────────────
    st.subheader("3.6  `compute_display_saddle()` — defined but not called")
    st.markdown(r"""
    The file also contains a `compute_display_saddle(alpha, rho, delta)` function
    that traces the exact saddle path via **backward integration**. It is
    `@st.cache_data`-decorated and fully implemented, but **not called** anywhere
    in the current app — the phase diagram does not display a saddle-path curve.

    The approach it implements:
    - Computes the stable-manifold slope $\mu = \lambda_2$ from the Jacobian eigenvalues.
    - Starts just off the steady state at $(\tilde k \pm \varepsilon,\; \tilde c \pm \varepsilon\mu)$.
    - Negates the ODE right-hand side so that integrating *forward* in pseudo-time
      moves *backward* along the actual trajectory — making the stable manifold
      attracting and easy to follow numerically.
    - Concatenates the left arm ($k < \tilde k$) and right arm ($k > \tilde k$) into
      a single saddle-path curve.

    If you want to re-enable it, call it before `make_phase_fig` and pass the
    result as an extra trace.
    """)

    # ── 3.7 Session state and widget sync ────────────────────────────────────
    st.subheader("3.7  Session state and widget synchronisation")
    st.markdown(r"""
    Streamlit reruns the entire script from top to bottom on every user
    interaction. This creates a challenge: widgets cannot have their value
    changed *after* they are instantiated (that would raise a
    `StreamlitAPIException`).

    The app resolves this with a **pending-value pattern**:

    1. When the bisection animation finishes, it stores the found $c(0)$ in
       `st.session_state["_c0_pending"]` and calls `st.rerun()`.
    2. At the **very top** of the script — before any widget is instantiated —
       it checks for `_c0_pending` and pre-populates the three widget keys
       (`c0_direct_val`, `c0_direct_slider`, `c0_direct_num`).
    3. When the widgets are then created, they read their pre-populated values
       and display them correctly.
    """)
    st.code("""
# ── At the very top of ramsey.py, before any st.slider / st.number_input ──
if "_c0_pending" in st.session_state:
    _pending = float(st.session_state.pop("_c0_pending"))
    st.session_state["c0_direct_val"]    = _pending
    st.session_state["c0_direct_slider"] = _pending
    st.session_state["c0_direct_num"]    = _pending
""", language="python")

    # ── 3.8 Animated bisection ───────────────────────────────────────────────
    st.subheader("3.8  Animated bisection")
    st.markdown(r"""
    Clicking **Find saddle path** sets `_run_bisection_anim = True` in session
    state. The animation loop (up to 40 iterations):

    1. Computes the midpoint $c^* = (\ell + h)/2$.
    2. Calls `shoot(c^*, ...)` to classify the trajectory.
    3. Appends the trajectory to a growing list `_trace_data`.
    4. Re-renders the Plotly figure in-place using `st.empty()` containers.
    5. Updates the bracket and **breaks early** if the trajectory converges.
    6. After the loop, calls `find_c0(..., n_iters=200)` for machine-precision
       accuracy (the animation is educational, not the precision source).
    7. Stores everything in `st.session_state["_anim_result"]` and calls
       `st.rerun()` so the main charts update.

    On rerun, the `elif "_anim_result" in st.session_state:` branch rebuilds the
    bisection diagram from the stored data — making it **persistent** across
    reruns without re-running the animation.
    """)

    # ── 3.9 Numerical parameters and tolerances ───────────────────────────────
    st.subheader("3.9  Numerical choices and tolerances")
    st.markdown("""
    | Setting | Value | Reason |
    |---------|-------|--------|
    | ODE method | RK45 | Explicit Runge-Kutta 4(5); good for smooth ODEs |
    | `max_step` | 0.05 | Prevents the adaptive stepper from taking enormous steps near the SS where the RHS is near zero |
    | Bisection iterations (visual) | ≤ 40 | Enough to demonstrate convergence visually; stops early on first converge |
    | Bisection iterations (precision) | 200 | Exceeds double-precision limits; result is cached by `@st.cache_data` |
    | `EXPLODE` threshold | 3.5 × k̃ | Empirically safe — far enough above k̃ to catch runaway paths |
    | `CLOSE_TOL` | 0.01 | 1 % of steady-state values; loose enough to stop `shoot()` quickly |
    | `CONV_TOL` | 0.02 | 2 % — slightly looser threshold used by `build_paths()` to mark convergence time t* |
    """)
