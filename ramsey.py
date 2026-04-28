import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp

st.set_page_config(layout="wide")
st.title("Ramsey–Cass–Koopmans Model")

# ── Constants ─────────────────────────────────────────────────────────────────
T_SHOOT   = 600
MAX_STEP  = 0.05
EXPLODE   = 3.5
CLOSE_TOL = 0.01
T_PATH    = 200
N_EVAL    = 300

# ── Apply any pending c(0) update BEFORE widgets are instantiated ─────────────
if "_c0_pending" in st.session_state:
    _pending = float(st.session_state.pop("_c0_pending"))
    st.session_state["c0_direct_val"]    = _pending
    st.session_state["c0_direct_slider"] = _pending
    st.session_state["c0_direct_num"]    = _pending

# ── Sidebar ──────────────────────────────────────────────────────────────────
# Pre-compute dynamic c(0) slider range from session state (before sidebar renders)
_alpha_v = float(st.session_state.get("alpha_val", 0.30))
_rho_v   = float(st.session_state.get("rho_val",  0.05))
_delta_v = float(st.session_state.get("delta_val", 0.00))
_k0_v    = float(st.session_state.get("k0_val",   0.238912227798098))
try:
    _k_ss_v = (_alpha_v / (_rho_v + _delta_v)) ** (1.0 / (1.0 - _alpha_v))
    _c_ss_v = _k_ss_v**_alpha_v - _delta_v * _k_ss_v
except Exception:
    _k_ss_v, _c_ss_v = 13.0, 2.2
_k0_max_dyn  = float(max(3.0 * _k_ss_v, 1.0))
_c0_max_dyn  = float(max(_c_ss_v * 2.5, _k0_v**_alpha_v * 4.0, 1.0))
# Seed: 40 % of output at k(0) — typically below saddle path → explodes visually
_c0_seed_dyn = float(max(0.001, min(_k0_v**_alpha_v * 0.4, _c0_max_dyn * 0.9)))

with st.sidebar:
    st.header("Parameters")

    def slider_with_input(label, min_val, max_val, default, step, key):
        # Initialise all three session-state entries on first run
        if f"{key}_val" not in st.session_state:
            st.session_state[f"{key}_val"]    = float(default)
        if f"{key}_slider" not in st.session_state:
            st.session_state[f"{key}_slider"] = float(default)
        if f"{key}_num" not in st.session_state:
            st.session_state[f"{key}_num"]    = float(default)

        def _on_slider():
            v = st.session_state[f"{key}_slider"]
            st.session_state[f"{key}_val"] = v
            st.session_state[f"{key}_num"] = v          # push to number input

        def _on_num():
            v = float(st.session_state[f"{key}_num"])
            st.session_state[f"{key}_val"] = v
            # clamp to slider range so the slider widget stays valid
            clamped = max(float(min_val), min(float(max_val), v))
            st.session_state[f"{key}_slider"] = clamped  # push to slider

        col_s, col_n = st.columns([3, 2])
        with col_s:
            st.slider(
                label, float(min_val), float(max_val),
                step=float(step), key=f"{key}_slider",
                on_change=_on_slider,
            )
        with col_n:
            st.number_input(
                "value",
                min_value=float(min_val), max_value=float(max_val),
                step=float(step), format="%.15g",
                key=f"{key}_num", label_visibility="hidden",
                on_change=_on_num,
            )
        return float(st.session_state[f"{key}_val"])

    alpha   = slider_with_input("α  (capital share)",     0.10, 0.60, 0.30,  0.01,  "alpha")
    rho     = slider_with_input("ρ  (time preference)",   0.01, 0.15, 0.05,  0.005, "rho")
    delta   = slider_with_input("δ  (depreciation)",      0.00, 0.15, 0.00,  0.005, "delta")

    st.divider()
    k0_input = slider_with_input("k(0)  (initial capital)", 0.001, _k0_max_dyn, 0.238912227798098, 0.001, "k0")

    st.divider()
    st.caption("Drag c(0) until the trajectory converges, or click below.")
    c0_direct = slider_with_input("c(0)  (initial consumption)", 0.001, _c0_max_dyn, _c0_seed_dyn, 0.001, "c0_direct")

    if st.button("🎯 Find saddle path", use_container_width=True,
                  help="Runs animated bisection to find the c(0) that converges to steady state."):
        st.session_state["_run_bisection_anim"] = True

    st.divider()
    if st.button("📐 Show/Hide derivations", use_container_width=True):
        st.session_state["_show_deriv"] = not st.session_state.get("_show_deriv", False)
    show_derivations = st.session_state.get("_show_deriv", False)

# ── Steady state ─────────────────────────────────────────────────────────────
k_ss = (alpha / (rho + delta)) ** (1.0 / (1.0 - alpha))
c_ss = k_ss**alpha - delta * k_ss
k0   = k0_input

# ── ODE system ───────────────────────────────────────────────────────────────
def make_rhs(alpha_, delta_, rho_):
    def rhs(t, y):
        k, c = y
        dkdt = k**alpha_ - delta_ * k - c
        dcdt = c * (alpha_ * k**(alpha_ - 1.0) - delta_ - rho_)
        return [dkdt, dcdt]
    return rhs

rhs = make_rhs(alpha, delta, rho)

# ── Shoot one trajectory and return (k_arr, c_arr, status) ───────────────────
def shoot(c0_candidate, k0_, k_ss_, c_ss_, rhs_, t_max=None):
    t_end = t_max if t_max is not None else T_SHOOT

    def ev_explode(t, y):
        return y[0] - EXPLODE * k_ss_
    ev_explode.terminal  = True
    ev_explode.direction = 1

    def ev_collapse(t, y):
        return min(y[0], y[1]) - 1e-10
    ev_collapse.terminal  = True
    ev_collapse.direction = -1

    def ev_converge(t, y):
        k, c = y
        return np.sqrt(((k - k_ss_) / k_ss_)**2 + ((c - c_ss_) / c_ss_)**2) - CLOSE_TOL
    ev_converge.terminal  = True
    ev_converge.direction = -1

    sol = solve_ivp(
        rhs_,
        (0, t_end),
        [k0_, c0_candidate],
        method="RK45",
        max_step=MAX_STEP,
        events=[ev_explode, ev_collapse, ev_converge],
        dense_output=False,
    )
    k_arr, c_arr = sol.y
    # Check terminal events in order: converge, explode, collapse
    if sol.t_events[2].size > 0:
        return k_arr, c_arr, "converge"
    if sol.status == 1:
        if sol.t_events[0].size > 0:
            return k_arr, c_arr, "explode"
        if sol.t_events[1].size > 0:
            return k_arr, c_arr, "collapse"
    if k_arr[-1] > EXPLODE * k_ss_:
        return k_arr, c_arr, "explode"
    if k_arr[-1] <= 1e-8 or c_arr[-1] <= 1e-8:
        return k_arr, c_arr, "collapse"
    return k_arr, c_arr, "converge"

# ── Cached bisection (n_iters controls precision) ───────────────────────────
@st.cache_data
def find_c0(alpha_, rho_, delta_, k0_abs_, n_iters=200):
    k_ss_ = (alpha_ / (rho_ + delta_)) ** (1.0 / (1.0 - alpha_))
    c_ss_ = k_ss_**alpha_ - delta_ * k_ss_
    k0_   = k0_abs_
    rhs_  = make_rhs(alpha_, delta_, rho_)

    lo = 1e-6
    hi = c_ss_ * 2.5
    for _ in range(n_iters):
        mid = 0.5 * (lo + hi)
        _, _, status = shoot(mid, k0_, k_ss_, c_ss_, rhs_)
        if status == "explode":
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

# ── Validate: final 20 % of time span must stay within 5 % of ss ─────────────
def validate_paths(t_eval_, k_p_, c_p_, k_ss_, c_ss_):
    cutoff = int(len(t_eval_) * 0.80)
    dist = np.sqrt(
        ((k_p_[cutoff:] - k_ss_) / k_ss_)**2 +
        ((c_p_[cutoff:] - c_ss_) / c_ss_)**2
    )
    return bool(np.all(dist < 0.05))

# ── Programmatic nullcline sanity check ───────────────────────────────────────
def assert_nullcline_intersection(k_ss_, c_ss_, alpha_, delta_):
    """Verify k-nullcline passes through (k_ss, c_ss). Raises if not."""
    residual = abs(k_ss_**alpha_ - delta_ * k_ss_ - c_ss_)
    if residual >= 1e-6:
        st.error(
            f"⚠️ Nullcline sanity check failed: "
            f"|k̃^α − δ·k̃ − c̃| = {residual:.2e} ≥ 1e-6. "
            "Check parameter consistency."
        )

# ── Build path arrays ─────────────────────────────────────────────────────────
# Returns (t_eval, k_path, c_path, t_conv, status) for any c(0).
# When c(0) is the saddle-path value the trajectory converges; otherwise it
# explodes (k → ∞) or collapses (k or c → 0).
CONV_TOL = 0.02

def build_paths(c0_, k0_, alpha_, delta_, rho_, k_ss_, c_ss_):
    rhs_ = make_rhs(alpha_, delta_, rho_)

    def ev_converged(t, y):
        k, c = y
        return np.sqrt(((k - k_ss_) / k_ss_)**2 + ((c - c_ss_) / c_ss_)**2) - CONV_TOL
    ev_converged.terminal  = True
    ev_converged.direction = -1

    def ev_explode(t, y):
        return y[0] - EXPLODE * k_ss_
    ev_explode.terminal  = True
    ev_explode.direction = 1

    def ev_collapse(t, y):
        return min(y[0], y[1]) - 1e-6
    ev_collapse.terminal  = True
    ev_collapse.direction = -1

    sol = solve_ivp(
        rhs_, (0, T_PATH), [k0_, c0_],
        method="RK45", max_step=MAX_STEP,
        events=[ev_converged, ev_explode, ev_collapse], dense_output=True,
    )

    # Determine outcome
    if sol.t_events[0].size > 0:
        status = "converge"
        t_end  = float(sol.t_events[0][0])
    elif sol.t_events[1].size > 0:
        status = "explode"
        t_end  = float(sol.t_events[1][0])
    elif sol.t_events[2].size > 0:
        status = "collapse"
        t_end  = float(sol.t_events[2][0])
    else:
        status = "timeout"
        t_end  = T_PATH

    t_conv = t_end if status == "converge" else float("nan")

    t_dense = np.linspace(0, t_end, N_EVAL)
    y_dense = sol.sol(t_dense)
    k_p = y_dense[0]
    c_p = y_dense[1]

    return t_dense, k_p, c_p, t_conv, status

# ── Plot bounds ──────────────────────────────────────────────────────────────
PHASE_K_CAP = 2.0   # multiples of k_ss  (≥ k_gold/k_ss for typical δ≥0.05)
PHASE_C_CAP = 1.6   # multiples of c_ss

# ── Full stable manifold for display ────────────────────────────────────────
# Uses backward integration from a point just off the steady state along the
# stable eigenvector — no bisection, no risk of landing on the unstable arm.
#
# Left arm  (k < k*): perturb (k*-eps, c*-eps*mu) and integrate backward -> k decreases
# Right arm (k > k*): perturb (k*+eps, c*+eps*mu) and integrate backward -> k increases
# mu = stable-manifold slope at SS = rho - lam1 = lam2 > 0
@st.cache_data
def compute_display_saddle(alpha_, rho_, delta_):
    k_ss_ = (alpha_ / (rho_ + delta_)) ** (1.0 / (1.0 - alpha_))
    c_ss_ = k_ss_**alpha_ - delta_ * k_ss_

    # Jacobian at SS: J11=rho, J12=-1, J21=c_ss*alpha*(alpha-1)*k_ss^(alpha-2), J22=0
    J21   = c_ss_ * alpha_ * (alpha_ - 1.0) * k_ss_ ** (alpha_ - 2.0)
    det_J = J21                             # < 0  (saddle confirmed)
    discr = rho_**2 - 4.0 * det_J          # > 0  (real eigenvalues)
    lam1  = 0.5 * (rho_ - np.sqrt(discr))  # stable eigenvalue (< 0)
    mu    = rho_ - lam1                     # stable-manifold slope = lam2 > 0

    rhs_ = make_rhs(alpha_, delta_, rho_)

    def rhs_back(t, y):
        f = rhs_(t, y)
        return [-f[0], -f[1]]

    eps = k_ss_ * 1e-4

    # Left arm: integrate backward from (k_ss - eps, c_ss - eps*mu)
    def ev_kmin(t, y): return y[0] - k_ss_ * 0.001
    ev_kmin.terminal = True; ev_kmin.direction = -1
    def ev_cmin(t, y): return y[1] - 1e-6
    ev_cmin.terminal = True; ev_cmin.direction = -1

    sol_l = solve_ivp(
        rhs_back, (0, 5_000), [k_ss_ - eps, c_ss_ - eps * mu],
        method="RK45", rtol=1e-9, atol=1e-11,
        events=[ev_kmin, ev_cmin], dense_output=True,
    )
    t_l  = np.linspace(0, float(sol_l.t[-1]), 800)
    y_l  = sol_l.sol(t_l)
    k_left = y_l[0][::-1]
    c_left = y_l[1][::-1]

    # Right arm: integrate backward from (k_ss + eps, c_ss + eps*mu)
    def ev_kmax(t, y): return y[0] - k_ss_ * PHASE_K_CAP
    ev_kmax.terminal = True; ev_kmax.direction = 1
    def ev_cmax(t, y): return y[1] - c_ss_ * PHASE_C_CAP
    ev_cmax.terminal = True; ev_cmax.direction = 1

    sol_r = solve_ivp(
        rhs_back, (0, 5_000), [k_ss_ + eps, c_ss_ + eps * mu],
        method="RK45", rtol=1e-9, atol=1e-11,
        events=[ev_kmax, ev_cmax], dense_output=True,
    )
    t_r  = np.linspace(0, float(sol_r.t[-1]), 200)
    y_r  = sol_r.sol(t_r)
    k_right = y_r[0]
    c_right = y_r[1]

    k_s = np.concatenate([k_left, [k_ss_], k_right])
    c_s = np.concatenate([c_left, [c_ss_], c_right])
    k_s = np.clip(k_s, 0, k_ss_ * PHASE_K_CAP)
    c_s = np.clip(c_s, 0, c_ss_ * PHASE_C_CAP)
    return k_s, c_s
# ── Nullcline grid ────────────────────────────────────────────────────────────
def nullcline_grid(k_ss_, alpha_, delta_):
    k_max   = k_ss_ * PHASE_K_CAP
    k_grid  = np.linspace(1e-6, k_max, 500)
    c_knull = k_grid**alpha_ - delta_ * k_grid
    # Only keep the physically meaningful (non-negative) portion
    mask    = c_knull >= 0
    return k_grid[mask], c_knull[mask]

# ── Plotly phase diagram ────────────────────────────────────────────
# k_traj, c_traj: actual trajectory the user's economy traverses
def make_phase_fig(k_ss_, c_ss_, k0_, c0_, alpha_, delta_,
                   rho_=None, k_traj=None, c_traj=None, extra_traces=None,
                   traj_status="converge"):
    k_grid, c_knull = nullcline_grid(k_ss_, alpha_, delta_)

    fig = go.Figure()

    # k-nullcline
    fig.add_trace(go.Scatter(
        x=k_grid, y=c_knull,
        mode="lines", line=dict(color="black", dash="dash", width=1.5),
        name="k-nullcline (dk/dt=0)",
        hovertemplate="k = %{x:.4f}<br>c = %{y:.4f}<extra>k-nullcline</extra>",
    ))

    # c-nullcline (vertical at k_ss, capped at plot y-limit)
    c_range = [0, c_ss_ * PHASE_C_CAP]
    fig.add_trace(go.Scatter(
        x=[k_ss_, k_ss_], y=c_range,
        mode="lines", line=dict(color="gray", dash="dot", width=1.5),
        name="c-nullcline (k=k̃)",
        hovertemplate="k = %{x:.4f}<extra>c-nullcline</extra>",
    ))

    # Extra animated trajectories
    if extra_traces:
        for tr in extra_traces:
            fig.add_trace(tr)

    # Actual user trajectory — colour signals outcome
    if k_traj is not None and c_traj is not None:
        k_cap = k_ss_ * PHASE_K_CAP
        c_cap = c_ss_ * PHASE_C_CAP
        kp = np.clip(k_traj, 0, k_cap)
        cp = np.clip(c_traj, 0, c_cap)
        traj_color_map = {"converge": "deepskyblue", "explode": "crimson",
                          "collapse": "darkorange", "timeout": "gray"}
        traj_color = traj_color_map.get(traj_status, "deepskyblue")
        fig.add_trace(go.Scatter(
            x=kp, y=cp,
            mode="lines", line=dict(color=traj_color, width=3, dash="solid"),
            name="Trajectory from k(0)",
            hovertemplate="k = %{x:.4f}<br>c = %{y:.4f}<extra>Trajectory</extra>",
        ))

    # Steady state — annotation dot (non-hoverable marker layer)
    fig.add_trace(go.Scatter(
        x=[k_ss_], y=[c_ss_],
        mode="markers+text",
        marker=dict(color="red", size=10, symbol="circle"),
        text=["steady state"],
        textposition="top right",
        name="Steady state",
        hoverinfo="skip",
    ))

    # Start point
    fig.add_trace(go.Scatter(
        x=[k0_], y=[c0_],
        mode="markers+text",
        marker=dict(color="green", size=10, symbol="circle"),
        text=["start"],
        textposition="top right",
        name="Start",
        hoverinfo="skip",
    ))

    fig.update_layout(
        title="Phase diagram",
        xaxis_title="k", yaxis_title="c",
        hovermode="closest",
        xaxis=dict(
            range=[0, k_ss_ * PHASE_K_CAP],
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikecolor="#888", spikethickness=1, spikedash="dash",
        ),
        yaxis=dict(range=[0, c_ss_ * PHASE_C_CAP]),
        legend=dict(font=dict(size=11)),
        margin=dict(l=50, r=20, t=50, b=50),
    )
    return fig

# ── Plotly time paths (stacked subplots) ──────────────────────────────────────
def make_time_fig(t_eval, k_path, c_path, k_ss_, c_ss_, k0_, c0_, t_conv=None, status="converge"):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Capital  k(t)", "Consumption  c(t)"),
    )

    # Curve colour: blue for saddle path, red for explode, orange for collapse
    color_map = {"converge": "royalblue", "explode": "crimson", "collapse": "darkorange", "timeout": "gray"}
    k_color   = color_map.get(status, "royalblue")
    c_color   = color_map.get(status, "crimson") if status == "converge" else k_color

    x_max = t_eval[-1] * 1.05

    # ── k(t) panel (top) ────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=t_eval, y=k_path,
        mode="lines", line=dict(color=k_color, width=2.5),
        name="k(t)",
        hovertemplate="t=%{x:.2f}<br>k=%{y:.4f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[0, x_max], y=[k_ss_, k_ss_],
        mode="lines", line=dict(color="gray", dash="dash", width=1),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)

    # ── c(t) panel (bottom) ─────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=t_eval, y=c_path,
        mode="lines", line=dict(color=c_color, width=2.5),
        name="c(t)",
        hovertemplate="t=%{x:.2f}<br>c=%{y:.4f}<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[0, x_max], y=[c_ss_, c_ss_],
        mode="lines", line=dict(color="gray", dash="dash", width=1),
        showlegend=False, hoverinfo="skip",
    ), row=2, col=1)

    # Convergence vertical line
    if t_conv is not None and not np.isnan(t_conv):
        for row in [1, 2]:
            fig.add_vline(x=t_conv, row=row, col=1,
                          line=dict(color="gray", dash="dot", width=1))
        fig.add_annotation(
            x=t_conv, y=k_ss_, xref="x", yref="y",
            text=f"t*={t_conv:.1f}",
            showarrow=False, font=dict(size=10, color="gray"),
            xanchor="left", xshift=4, yanchor="bottom",
        )

    # SS reference annotations
    fig.add_annotation(x=x_max * 0.98, y=k_ss_, xref="x", yref="y",
        text=f"k̃={k_ss_:.2f}", showarrow=False,
        font=dict(size=10, color="gray"), xanchor="right", yanchor="bottom", yshift=2)
    fig.add_annotation(x=x_max * 0.98, y=c_ss_, xref="x2", yref="y2",
        text=f"c̃={c_ss_:.2f}", showarrow=False,
        font=dict(size=10, color="gray"), xanchor="right", yanchor="bottom", yshift=2)

    # Y-axis: let plotly auto-range so exploding trajectories are visible
    fig.update_xaxes(range=[0, x_max], title_text="Time  t", row=2, col=1)
    fig.update_xaxes(range=[0, x_max], row=1, col=1)
    fig.update_yaxes(title_text="k", row=1, col=1)
    fig.update_yaxes(title_text="c", row=2, col=1)

    fig.update_layout(
        height=500,
        hovermode="x unified",
        showlegend=False,
        margin=dict(l=50, r=20, t=30, b=40),
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# ── Main solve ────────────────────────────────────────────────────────────────
assert_nullcline_intersection(k_ss, c_ss, alpha, delta)

# c(0) is whatever the user set on the slider
c0 = c0_direct

t_eval, k_path, c_path, t_conv, traj_status = build_paths(c0, k0, alpha, delta, rho, k_ss, c_ss)

conv_time = t_conv

# Saddle path curve no longer pre-computed for display

# ── Metrics ───────────────────────────────────────────────────────────────────
status_label = {"converge": "✅ converged", "explode": "💥 exploded", "collapse": "📉 collapsed", "timeout": "⏱ timed out"}
m1, m2, m3, m4 = st.columns(4)
m1.metric("k̃", f"{k_ss:.6g}")
m2.metric("c̃", f"{c_ss:.6g}")
m3.metric("Convergence time t*", f"{conv_time:.1f}" if traj_status == "converge" else "—")
m4.metric("Outcome", status_label[traj_status])

# ── Main charts ───────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    fig_phase = make_phase_fig(k_ss, c_ss, k0, c0, alpha, delta,
                               rho_=rho, k_traj=k_path, c_traj=c_path,
                               traj_status=traj_status)
    st.plotly_chart(fig_phase, use_container_width=True)

with col2:
    fig_time = make_time_fig(t_eval, k_path, c_path, k_ss, c_ss, k0, c0, t_conv=conv_time, status=traj_status)
    st.plotly_chart(fig_time, use_container_width=True)

# ── Animated bisection ────────────────────────────────────────────────────────
_run_anim = st.session_state.pop("_run_bisection_anim", False)
_COLOR_A = {
    "explode":  "rgba(220, 50,  50, 0.6)",
    "collapse": "rgba(255,140,   0, 0.6)",
    "converge": "rgba( 50,180,  50, 0.85)",
    "saddle":   "deepskyblue",
}
_WIDTH_A = {"explode": 1.4, "collapse": 1.4, "converge": 1.4, "saddle": 3}

def _build_anim_fig(trace_data, k_ss_, c_ss_, k0_, c0_, alpha_, delta_, rho_):
    """Rebuild the phase figure with all bisection traces overlaid."""
    extra = [
        go.Scatter(
            x=kk, y=cc, mode="lines",
            line=dict(color=_COLOR_A.get(ss, "gray"), width=_WIDTH_A.get(ss, 1.4)),
            showlegend=False, hoverinfo="skip",
        )
        for kk, cc, ss in trace_data
    ]
    return make_phase_fig(k_ss_, c_ss_, k0_, c0_, alpha_, delta_,
                          rho_=rho_, extra_traces=extra)

if _run_anim:
    # Clear any previous result so the section re-renders fresh
    st.session_state.pop("_anim_result", None)
    st.divider()
    _anim_status = st.empty()
    _anim_chart  = st.empty()
    _lo_a, _hi_a = 1e-6, c_ss * 2.5
    _trace_data: list = []   # list of (k_list, c_list, status_str)
    _ANIM_ITERS  = 40

    for _it in range(_ANIM_ITERS):
        _mid_a = 0.5 * (_lo_a + _hi_a)
        _k_a, _c_a, _stat_a = shoot(_mid_a, k0, k_ss, c_ss, rhs)
        _trace_data.append((
            np.clip(_k_a, 0, k_ss * PHASE_K_CAP).tolist(),
            np.clip(_c_a, 0, c_ss * PHASE_C_CAP).tolist(),
            _stat_a,
        ))
        _anim_chart.plotly_chart(
            _build_anim_fig(_trace_data, k_ss, c_ss, k0, c0, alpha, delta, rho),
            use_container_width=True,
        )
        _label_a = {"explode": "💥 exploded", "collapse": "📉 collapsed",
                    "converge": "✅ converged"}[_stat_a]
        _anim_status.markdown(
            f"**Bisection iter {_it + 1}/{_ANIM_ITERS}** — "
            f"c(0) = `{_mid_a:.10g}` — {_label_a} — "
            f"bracket: [`{_lo_a:.8g}`, `{_hi_a:.8g}`]"
        )
        if _stat_a == "explode":
            _lo_a = _mid_a
        else:
            _hi_a = _mid_a
        time.sleep(0.05)
        if _stat_a == "converge":
            break

    _c0_anim = float(0.5 * (_lo_a + _hi_a))  # 20-iter visual result

    # High-precision result via cached 200-iteration bisection
    _anim_status.markdown("*Refining precision…*")
    _c0_precise = find_c0(alpha, rho, delta, k0, n_iters=200)
    _c0_clamped = max(0.001, min(_c0_max_dyn, _c0_precise))

    # Append the final saddle-path trajectory in cyan
    _t_sp, _k_sp, _c_sp, _, _ = build_paths(_c0_clamped, k0, alpha, delta, rho, k_ss, c_ss)
    _trace_data.append((
        np.clip(_k_sp, 0, k_ss * PHASE_K_CAP).tolist(),
        np.clip(_c_sp, 0, c_ss * PHASE_C_CAP).tolist(),
        "saddle",
    ))
    _anim_chart.plotly_chart(
        _build_anim_fig(_trace_data, k_ss, c_ss, k0, _c0_clamped, alpha, delta, rho),
        use_container_width=True,
    )
    _anim_status.success(f"✅ Saddle path found — c(0) = {_c0_precise:.15g}")

    # Persist result so it survives the upcoming rerun
    st.session_state["_anim_result"] = {
        "traces":   _trace_data,
        "c0_found": _c0_precise,
        "k0":       k0,
        "alpha":    alpha, "delta": delta, "rho": rho,
        "k_ss":     k_ss,  "c_ss":  c_ss,
    }
    time.sleep(1.0)
    # Update slider to the found value and redraw main charts
    st.session_state["_c0_pending"] = _c0_clamped
    st.rerun()

# ── Persistent bisection diagram (shown after animation rerun) ────────────────
elif "_anim_result" in st.session_state:
    _r = st.session_state["_anim_result"]
    st.divider()
    st.success(f"✅ Saddle path found — c(0) = {_r['c0_found']:.15g}")
    st.plotly_chart(
        _build_anim_fig(
            _r["traces"], _r["k_ss"], _r["c_ss"],
            _r["k0"], _r["c0_found"],
            _r["alpha"], _r["delta"], _r["rho"],
        ),
        use_container_width=True,
    )

# ── Derivations ───────────────────────────────────────────────────────────────
if show_derivations:
    st.divider()
    st.header("Model derivations")

    # Precompute Jacobian values for display
    # J11 = ∂(dk/dt)/∂k = α·k̃^(α-1) - δ  (= ρ at SS by the dc/dt=0 condition)
    J11 = alpha * k_ss**(alpha - 1.0) - delta  # = ρ at steady state
    J12 = -1.0
    J21 = c_ss * alpha * (alpha - 1.0) * k_ss**(alpha - 2.0)
    J22 = alpha * k_ss**(alpha - 1.0) - delta - rho   # = 0 at steady state
    det_J = J11 * J22 - J12 * J21
    tr_J  = J11 + J22
    discr = tr_J**2 - 4.0 * det_J
    lam1  = 0.5 * (tr_J - np.sqrt(discr))
    lam2  = 0.5 * (tr_J + np.sqrt(discr))

    st.markdown("**Household objective:** maximise discounted log utility subject to the capital accumulation equation.")
    st.latex(rf"\max_{{c(t)}} \int_0^\infty e^{{-\rho t}} \ln c(t)\, dt \qquad \rho = {rho}")

    st.markdown("**Capital and Euler equations** (the two ODEs of the system):")
    st.latex(rf"\dot{{k}}(t) = k(t)^{{\alpha}} - \delta\, k(t) - c(t) \qquad \alpha={alpha},\; \delta={delta}")
    st.latex(rf"\dot{{c}}(t) = c(t)\bigl[\alpha\, k(t)^{{\alpha-1}} - \delta - \rho\bigr]")

    st.markdown(r"**Steady state:** set $\dot{k}=0$ and $\dot{c}=0$ and solve.")
    st.latex(rf"\tilde{{k}} = \left(\frac{{\alpha}}{{\rho+\delta}}\right)^{{\!\tfrac{{1}}{{1-\alpha}}}} "
            rf"= \left(\frac{{{alpha}}}{{{rho}+{delta}}}\right)^{{\!\tfrac{{1}}{{1-{alpha}}}}} = {k_ss:.6f}")
    st.latex(rf"\tilde{{c}} = \tilde{{k}}^\alpha - \delta\,\tilde{{k}} "
            rf"= {k_ss:.6f}^{{{alpha}}} - {delta}\times{k_ss:.6f} = {c_ss:.6f}")

    st.markdown("**Euler equation interpretation:** the household equates the marginal product of capital net of depreciation to the rate of time preference; any gap drives consumption growth or decline.")
    mpk_ss = alpha * k_ss**(alpha - 1.0)
    st.latex(rf"\frac{{\dot{{c}}}}{{c}} = \underbrace{{\alpha k^{{\alpha-1}}}}_{{\text{{MPK}}}} - \delta - \rho "
            rf"\;\;\Rightarrow\;\; \text{{at }}\tilde{{k}}:\; \text{{MPK}} = {mpk_ss:.6f},\;\; "
            rf"\delta+\rho = {delta + rho:.6f} \;\checkmark")

    st.markdown(r"**Linearised system around $(\tilde{k},\tilde{c})$:** let $\hat{k}=k-\tilde{k}$, $\hat{c}=c-\tilde{c}$. The Jacobian is:")
    st.latex(
        r"J = \begin{pmatrix}"
        r"\alpha\tilde{k}^{\alpha-1}-\delta & -1 \\"
        r"\tilde{c}\,\alpha(\alpha-1)\tilde{k}^{\alpha-2} & \alpha\tilde{k}^{\alpha-1}-\delta-\rho"
        r"\end{pmatrix}"
    )
    st.markdown(r"At the steady state $\alpha\tilde{k}^{\alpha-1}=\rho+\delta$, so $J_{11}=\rho$ and $J_{22}=0$:")
    st.latex(rf"J = \begin{{pmatrix}} {J11:.6f} & {J12:.1f} \\\\ {J21:.6f} & {J22:.2g} \end{{pmatrix}}"
            rf"\qquad \det J = {det_J:.6f},\;\; \mathrm{{tr}}\,J = {tr_J:.6f}")

    st.markdown(r"**Eigenvalues of $J$:** since $\det J < 0$, one eigenvalue is positive and one is negative — confirming the saddle-point structure.")
    _tr_str   = f"{tr_J:.6f}"
    _discr_str = f"{discr:.6f}"
    st.latex(
        rf"\lambda_{{1,2}} = \frac{{\mathrm{{tr}}(J) \pm \sqrt{{\mathrm{{tr}}(J)^2 - 4\det J}}}}{{2}}"
        rf" = \frac{{{_tr_str} \pm \sqrt{{{_discr_str}}}}}{{2}}"
    )
    st.latex(rf"\lambda_1 = {lam1:.6f} \;(\text{{stable}}),\qquad \lambda_2 = {lam2:.6f} \;(\text{{unstable}})"
             rf"\qquad [\text{{note: }} \mathrm{{tr}}(J) = J_{{11}}+J_{{22}} = {J11:.6f}+{J22:.2g} = {tr_J:.6f} = \rho]")
    st.markdown(rf"The **saddle path** is the one-dimensional stable manifold corresponding to $\lambda_1 = {lam1:.6f}$. "
                rf"Only the unique initial $c(0) = {c0:.6f}$ that places the economy on this manifold converges to "
                rf"$(\tilde{{k}},\tilde{{c}}) = ({k_ss:.4f},\, {c_ss:.4f})$.")
