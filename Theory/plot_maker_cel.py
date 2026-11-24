import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import source_twospike as sts       # theory
import source_plots as sp           # functions used to make the plots (simulations)
import source_style as ss           # style of the figures (report / presentation)
from class_twospike import cv_mu, mu_equilibrium
# The TwoSpikes class is in class_twospike.py


ALPHA = np.sqrt(0.5)
DIR_PLOTS = "results"       # la ou on envoie nos plots (quel folder)
os.makedirs(DIR_PLOTS, exist_ok=True)


# ---------------------------------------------------------------- Evaluation loss vs mu_R (methods A and B)
def loss_eval_vs_mu(n=500, lam1=3.0, lam2=2.0, rho=0.5, M=20, methods=("A",),
                    mu_values=np.geomspace(0.02, 5, 25), mode="report", raw=False, title=False):
    """Evaluation loss L_eval(x_hat(mu_R)) vs mu_R, mean +- std/sqrt(M) over M samples.
    L_eval = alpha ||Y1 - x x^T||_F^2 + sqrt(1-alpha^2) ||Y2 - x x^T||_F^2   (TwoSpikes.Loss_eval_2).
    (a) L_eval - const, same samples for all mu (cv_mu centered=True).
    (b) overlaps m1 (solid), m2 (dashed) of the estimator.
    raw=True: also a separate figure with the raw cv_mu (new samples per mu, very noisy).
    Vertical lines: mu_opt (dotted, color of the method) and mu_eq = lam2 / (2 (lam1-1)^2) (black dashed).
    mode = "report" or "presentation"."""
    ss.set_style(mode)
    mu_eq = mu_equilibrium(lam1, lam2)
    tag = f"lam1={lam1}_lam2={lam2}_rho={rho}"

    fig, (ax_L, ax_m) = ss.figure(ncols=2, aspect=0.85)
    if raw:
        fig_raw, ax_raw = ss.figure(width=0.6)

    for method in methods:
        st = ss.METHOD[method]
        c = st["color"]

        mu_opt, L, err, ov = cv_mu(mu_values, n, rho, lam1, lam2, M=M, method=method,
                                   centered=True, return_overlaps=True)
        ax_L.errorbar(mu_values, L, err, color=c, marker=st["marker"], label=st["label"])
        ss.vline(ax_L, mu_opt, color=c, ls=":", label=rf"$\mu^*_{{{method}}} = {mu_opt:.2f}$")
        
        ax_m.plot(mu_values, ov[:, 0], color=c, marker=st["marker"], ls="-")
        ax_m.plot(mu_values, ov[:, 1], color=c, marker=st["marker"], ls="--", mfc="white")
        ss.vline(ax_L, mu_opt, color=c, ls=":", label=rf"$\mu^*_{{{method}}} = {mu_opt:.2f}$")
        print(f"method {method}: mu_opt = {mu_opt:.4f}")

        if raw:
            _, s_raw, e_raw = cv_mu(mu_values, n, rho, lam1, lam2, M=M, method=method)
            ax_raw.errorbar(mu_values, s_raw, e_raw, color=c, marker=st["marker"], label=st["label"])

    # (a) loss
    ss.vline(ax_L, mu_eq, label=rf"$\mu_{{eq}} = {mu_eq:.2f}$")
    ax_L.set_xscale("log")
    ax_L.set_xlabel(r"$\mu_R$")
    ax_L.set_ylabel(r"$\mathcal{L}_{\mathrm{eval}} - \mathrm{const}$")
    ax_L.legend(loc="upper right")

    # (b) overlaps: legend = line style (which signal) + color/marker (which method)
    ss.vline(ax_m, mu_eq)
    ax_m.set_xscale("log")
    ax_m.set_xlabel(r"$\mu_R$")
    ax_m.set_ylabel(r"overlap $m_l = |\langle \hat{x}, x_l \rangle|$")
    ax_m.set_ylim(0, 1)
    #handles = [plt.Line2D([], [], color=ss.INK, ls="-", label=r"$m_1$"),
    #           plt.Line2D([], [], color=ss.INK, ls="--", label=r"$m_2$")]
    handles = [plt.Line2D([], [], color=ss.METHOD[m]["color"], marker=ss.METHOD[m]["marker"],
                           ls="", label=ss.METHOD[m]["label"]) for m in methods]
    ax_m.legend(handles=handles, loc="lower right", ncol=2)

    ss.panel_labels([ax_L, ax_m])
    if title:
        fig.suptitle(rf"$n={n}$, $\lambda_1={lam1}$, $\lambda_2={lam2}$, $\rho={rho}$, {M} samples")
    ss.save(fig, f"loss_eval_vs_mu_{tag}_{mode}", DIR_PLOTS)

    if raw:
        ss.vline(ax_raw, mu_eq, label=rf"$\mu_{{eq}} = {mu_eq:.2f}$")
        ax_raw.set_xscale("log")
        ax_raw.set_xlabel(r"$\mu_R$")
        ax_raw.set_ylabel(r"$\mathcal{L}_{\mathrm{eval}}$ (new samples per $\mu_R$)")
        ax_raw.legend()
        ss.save(fig_raw, f"loss_eval_raw_vs_mu_{tag}_{mode}", DIR_PLOTS)



# ---------------------------------------------------------------- optimality condition dL/dmu = 0 (method A)
def crossing(x, y):
    """x where y changes sign from - to + (linear interpolation, on log x). None if no crossing."""
    idx = np.where((y[:-1] < 0) & (y[1:] >= 0))[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    lx = np.log(x[i]) - y[i] * (np.log(x[i + 1]) - np.log(x[i])) / (y[i + 1] - y[i])
    return np.exp(lx)


def optimality_A(n=600, lam1=3.0, lam2=2.0, rho=0.5, M=30, p=1,
                 mu_values=np.geomspace(0.05, 3, 30), mode="report"):
    """Optimality of mu_R for method A.
    (a) loss vs mu:
        - theory   : L(mu) = -2 sum_l w_l lam_l^p m_l^2(mu), overlaps of section 9.2 (sts.loss_eval_A_theory)
        - numerical: simulated L_eval (TwoSpikes.Loss_eval_2) - const, same M samples for all mu
                     (cv_mu centered=True), mean +- std/sqrt(M)
    (b) derivative vs mu:
        - theory   : analytical form of eq. (11.4), dL/dmu = -2 sum_l w_l lam_l^p d(m_l^2)/dmu (sts.dloss_dmu_A_theory)
        - numerical: finite differences of the simulated loss of (a)
        stars = zero crossings = mu*.
    p = 1: weights lam_l (expansion of ||Y_l - x x^T||^2, same as the simulated loss).
    p = 2: weights lam_l^2 as in eq. (11.2) of the report (then theory and simulation do not compute the same loss).
    The simulated loss also contains the noise term -2 sum_l w_l x^T Z_l x, which the theory drops:
    it shifts the curve in (a) and moves mu* a little."""
    ss.set_style(mode)
    mu_eq = mu_equilibrium(lam1, lam2)

    # theory
    mu_th = np.geomspace(mu_values[0], mu_values[-1], 400)
    L_th = np.array([sts.loss_eval_A_theory(m, lam1, lam2, rho, ALPHA, p) for m in mu_th])
    dL_th = np.array([sts.dloss_dmu_A_theory(m, lam1, lam2, rho, ALPHA, p) for m in mu_th])
    mu_star_th = sts.mu_opt_A_theory(lam1, lam2, rho, ALPHA, p)

    # numerics: simulated loss and its finite-difference derivative
    _, L_num, L_err, _ = cv_mu(mu_values, n, rho, lam1, lam2, M=M, method="A",
                               centered=True, return_overlaps=True)
    dL_num = np.gradient(L_num, mu_values)
    mu_star_num = crossing(mu_values, dL_num)

    fig, (ax_L, ax_d) = ss.figure(ncols=2, aspect=0.85)
    c_th, c_num = ss.REF, ss.COLORS[0]

    # (a) the loss
    ax_L.plot(mu_th, L_th, color=c_th, label=r"theory ($n\to\infty$)")
    ax_L.errorbar(mu_values, L_num, L_err, color=c_num, marker="o", ls="", label=rf"simulation ($n={n}$)")
    ax_L.set_ylabel(r"$\mathcal{L}_{\mathrm{eval}}(\mu_R) - \mathrm{const}$")

    # (b) the optimality condition
    ax_d.axhline(0, color=ss.INK, lw=0.8)
    ax_d.plot(mu_th, dL_th, color=c_th)
    ax_d.plot(mu_values, dL_num, color=c_num, marker="o", ls="")
    ax_d.set_ylabel(r"$\partial \mathcal{L}_{\mathrm{eval}} / \partial \mu_R$")
    top = max(np.max(dL_th), np.max(dL_num), 1e-3)          # zoom on the zero crossing
    ax_d.set_ylim(-4 * top, 2 * top)

    # zero crossings
    for m, c, ls, name in [(mu_star_th, c_th, "-.", "th"), (mu_star_num, c_num, ":", "num")]:
        if m is None:
            continue
        ax_d.plot(m, 0, marker="*", ms=2.5 * plt.rcParams["lines.markersize"], color=c, mec="white", zorder=5)
        ss.vline(ax_d, m, color=c, ls=ls)
        ss.vline(ax_L, m, color=c, ls=ls, label=rf"$\mu^*_{{\mathrm{{{name}}}}} = {m:.2f}$")
        print(f"mu* {name:4s} = {m:.4f}")
    ss.vline(ax_L, mu_eq, color="#888888", ls="--", label=rf"$\mu_{{eq}} = {mu_eq:.2f}$")
    ss.vline(ax_d, mu_eq, color="#888888", ls="--")

    for ax in (ax_L, ax_d):
        ax.set_xscale("log")
        ax.set_xlabel(r"$\mu_R$")
    fig.legend(*ax_L.get_legend_handles_labels(), loc="outside lower center", ncol=5)
    ss.panel_labels([ax_L, ax_d])
    ss.save(fig, f"optimality_A_lam1={lam1}_lam2={lam2}_rho={rho}_p={p}_{mode}", DIR_PLOTS)


# ---------------------------------------------------------------- biased vs unbiased evaluation loss (section 11)
from class_twospike import TwoSpikes


def _sample_losses(S, xhat, Y1p, Y2p):
    """Simulated losses of one estimator, minus the x_hat-independent constants (exact, no approximation):
        ||Y - x x^T||^2 = ||Y||^2 + 1 - 2 x^T Y x   (||x|| = 1)
    biased  : -2 sum_l x^T Y_l  x   (same observations used to build x_hat)            eq. (11.1)
    unbiased: -2 sum_l x^T Y'_l x   (Y'_l = lam_l x_l x_l^T + Z'_l, fresh noise)          eq. (11.3)
    signal  : -2 sum_l lam_l m_l^2  (simulated overlaps)                                 eq. (11.4)"""
    xhat = xhat / np.linalg.norm(xhat)
    m1, m2 = abs(xhat @ S.x1), abs(xhat @ S.x2)
    L_b = -2 * (xhat @ S.Y1 @ xhat + xhat @ S.Y2 @ xhat)
    L_u = -2 * (xhat @ Y1p @ xhat + xhat @ Y2p @ xhat)
    L_s = -2 * (S.lam1 * m1 ** 2 + S.lam2 * m2 ** 2)
    return L_b, L_u, L_s


def _fresh(S):
    """New observations of the same signals x1, x2 (independent noise)."""
    return (S.lam1 * np.outer(S.x1, S.x1) + S.symmetric_gaussian_matrix(),
            S.lam2 * np.outer(S.x2, S.x2) + S.symmetric_gaussian_matrix())


def _plot_losses(ax, x, sims, x_th, th, xlabel, name):
    """sims = (L_b, L_u, L_s) of shape (M, len(x)); th = (L_eval, L_biased) on x_th.
    Each curve gets a vertical line at its own minimum."""
    curves = [
        (r"$\mathcal{L}_{\mathrm{biased}}$ (num.)",   sims[0], ss.COLORS[1], "s"),
        (r"$\mathcal{L}_{\mathrm{eval}}$, fresh noise (num.)", sims[1], ss.COLORS[0], "o"),
        (r"$-2\sum_l \lambda_l m_l^2$ (num.)",        sims[2], ss.COLORS[2], "^"),
    ]
    M = sims[0].shape[0]
    for lab, L, ls, vls in [(r"$-2\sum_l \lambda_l m_l^2$ (theory)", th[0], "-", "-."),
                            (r"$\mathcal{L}_{\mathrm{biased}}$ (theory)", th[1], "--", "--")]:
        if L is None:                                   # no theory available (biased loss of method B)
            continue
        xm = x_th[np.argmin(L)]
        ax.plot(x_th, L, color=ss.REF, ls=ls, label=rf"{lab}, ${name}_{{\min}} = {xm:.2f}$")
        ss.vline(ax, xm, color=ss.REF, ls=vls)
        print(f"  {lab:45s} min at {xm:.3f}")
    for lab, L, c, mk in curves:
        mean, err = L.mean(0), L.std(0) / np.sqrt(M)
        xm = x[np.argmin(mean)]
        ax.errorbar(x, mean, err, color=c, marker=mk, ls="", label=rf"{lab}, ${name}_{{\min}} = {xm:.2f}$")
        ss.vline(ax, xm, color=c, ls=":")
        print(f"  {lab:45s} min at {xm:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\mathcal{L} - \mathrm{const}$")


def _sims_naive(n, lam1, lam2, rho, M, alphas):
    """(3, M, len(alphas)) simulated losses of the naive estimator; same samples for every alpha."""
    sims = np.zeros((3, M, len(alphas)))
    for s in range(M):
        S = TwoSpikes(n, lam1, lam2, rho)
        Y1p, Y2p = _fresh(S)
        for i, al in enumerate(alphas):
            _, xhat = TwoSpikes.top_eigenvector(al * S.Y1 + np.sqrt(1 - al ** 2) * S.Y2)
            sims[:, s, i] = _sample_losses(S, xhat, Y1p, Y2p)
        print(f"  naive sample {s + 1}/{M}", flush=True)
    return sims


def _sims_fisher(method, n, lam1, lam2, rho, M, mu_values):
    """(3, M, len(mu_values)) simulated losses of the method A or B estimator; same samples for every mu."""
    sims = np.zeros((3, M, len(mu_values)))
    for s in range(M):
        S = TwoSpikes(n, lam1, lam2, rho)
        Y1p, Y2p = _fresh(S)
        for i, mu in enumerate(mu_values):
            S.set_mu(mu)
            S.compute_method([method])
            sims[:, s, i] = _sample_losses(S, S.get_theta(method), Y1p, Y2p)
        print(f"  {method} sample {s + 1}/{M}", flush=True)
    return sims


def loss_naive_vs_alpha(n=800, lam1=3.0, lam2=2.0, rho=0.5, M=20,
                        alphas=np.linspace(0.0, 1.0, 26), mode="report"):
    """Naive method: biased loss, unbiased loss (fresh noise), -2 sum lam_l m_l^2 (num. and theory)
    vs the importance factor alpha, with alpha_min of every curve.
    Same M samples (x_l, Z_l, Z'_l) for every alpha."""
    ss.set_style(mode)
    tag = f"lam1={lam1}_lam2={lam2}_rho={rho}"
    cache = os.path.join(DIR_PLOTS, f"sims_naive_vs_alpha_{tag}_n={n}_M={M}.npz")
    if os.path.exists(cache) and np.array_equal(np.load(cache)["x"], alphas):
        sims = np.load(cache)["sims"]
    else:
        sims = _sims_naive(n, lam1, lam2, rho, M, alphas)
        np.savez(cache, x=alphas, sims=sims)

    a_th = np.linspace(0, 1, 501)
    th = np.array([sts.losses_theory(*sts.naive_theory(a, lam1, lam2, rho), lam1, lam2) for a in a_th]).T

    print("naive, vs alpha:")
    fig, ax = ss.figure(width=0.7, aspect=1.0)
    _plot_losses(ax, alphas, sims, a_th, th, r"$\alpha$", r"\alpha")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=1)
    ss.save(fig, f"loss_biased_naive_vs_alpha_lam1={lam1}_lam2={lam2}_rho={rho}_{mode}", DIR_PLOTS)


def loss_fisher_vs_mu(method="A", n=800, lam1=3.0, lam2=2.0, rho=0.5, M=20,
                      mu_values=np.geomspace(0.02, 5, 26), mode="report"):
    """Method A or B: biased loss, unbiased loss (fresh noise), -2 sum lam_l m_l^2 (num. and theory)
    vs mu_R, with mu_R,min of every curve. Same M samples for every mu.
    Theory: A -> overlaps and noise terms of section 9 / appendix C.3;
            B -> overlaps of sts.overlap_B_theory only (no noise-term theory for B: no biased theory curve)."""
    ss.set_style(mode)
    tag = f"lam1={lam1}_lam2={lam2}_rho={rho}"
    cache = os.path.join(DIR_PLOTS, f"sims_{method}_vs_mu_{tag}_n={n}_M={M}.npz")
    if os.path.exists(cache) and np.array_equal(np.load(cache)["x"], mu_values):
        sims = np.load(cache)["sims"]
    else:
        sims = _sims_fisher(method, n, lam1, lam2, rho, M, mu_values)
        np.savez(cache, x=mu_values, sims=sims)

    mu_th = np.geomspace(mu_values[0], mu_values[-1], 501)
    if method == "A":
        th = np.array([sts.losses_theory(*sts.methodA_theory(m, lam1, lam2, rho), lam1, lam2) for m in mu_th]).T
    else:
        ov = np.array([sts.overlap_B_theory(lam1, lam2, rho, m, n)[:2] for m in mu_th])
        th = (-2 * (lam1 * ov[:, 0] ** 2 + lam2 * ov[:, 1] ** 2), None)

    print(f"method {method}, vs mu_R:")
    fig, ax = ss.figure(width=0.7, aspect=1.0)
    _plot_losses(ax, mu_values, sims, mu_th, th, r"$\mu_R$", r"{\mu_R}")
    ax.set_xscale("log")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=1)
    ss.save(fig, f"loss_biased_{method}_vs_mu_lam1={lam1}_lam2={lam2}_rho={rho}_{mode}", DIR_PLOTS)


# ---------------------------------------------------------------- tuned methods compared vs lam1, rho, lam2
from class_twospike import solve_norm_constrained

ALPHA_GRID = np.linspace(0.0, 1.0, 21)
MU_GRID = np.geomspace(0.02, 10, 21)


def _sims_tuning(n, lam1, lam2, rho, M, alphas=ALPHA_GRID, mus=MU_GRID):
    """Simulated (biased, unbiased) losses of the three methods on their parameter grid.
    Returns {"naive": (2, M, len(alphas)), "A": (2, M, len(mus)), "B": (2, M, len(mus))}.
    For one sample, theta^(1) is computed once and A(mu) diagonalized once for methods A and B."""
    out = {"naive": np.zeros((2, M, len(alphas))),
           "A": np.zeros((2, M, len(mus))), "B": np.zeros((2, M, len(mus)))}
    for s in range(M):
        S = TwoSpikes(n, lam1, lam2, rho)
        Y1p, Y2p = _fresh(S)
        for i, al in enumerate(alphas):
            _, x = TwoSpikes.top_eigenvector(al * S.Y1 + np.sqrt(1 - al ** 2) * S.Y2)
            out["naive"][:, s, i] = _sample_losses(S, x, Y1p, Y2p)[:2]
        th1 = S.compute_theta1()
        F1th1 = S.fisher_MS(th1, lam1) @ th1
        for i, mu in enumerate(mus):
            A = S.A(th1, mu)
            _, xA = TwoSpikes.top_eigenvector(A)
            xB, _ = solve_norm_constrained(A, 2 * mu * F1th1)
            out["A"][:, s, i] = _sample_losses(S, xA, Y1p, Y2p)[:2]
            out["B"][:, s, i] = _sample_losses(S, xB, Y1p, Y2p)[:2]
    return out


def _tuned_results(vary, values, n, lam1, lam2, rho, M):
    """Simulated losses of naive, A, B on their parameter grids, for every value of `vary` (cached in results/).
    Returns {method: array (len(values), 2, M, len(grid))}, index 1 of axis 1 = unbiased loss."""
    par = dict(lam1=lam1, lam2=lam2, rho=rho)
    tag = "_".join(f"{k}={v}" for k, v in par.items() if k != vary)
    cache = os.path.join(DIR_PLOTS, f"sims_tuned_vs_{vary}_{tag}_n={n}_M={M}.npz")
    if os.path.exists(cache) and np.array_equal(np.load(cache)["values"], values):
        return {m: np.load(cache)[m] for m in ("naive", "A", "B")}
    res = {m: [] for m in ("naive", "A", "B")}
    for k, v in enumerate(values):
        par[vary] = v
        out = _sims_tuning(n, par["lam1"], par["lam2"], par["rho"], M)
        for m in res:
            res[m].append(out[m])
        print(f"  {vary} = {v:.3f} done ({k + 1}/{len(values)})", flush=True)
    res = {m: np.array(r) for m, r in res.items()}
    np.savez(cache, values=values, **res)
    return res


def _tuned_theory(method, vary, values, n, lam1, lam2, rho):
    """Theoretical unbiased loss L_eval = -2 (lam1 m1^2 + lam2 m2^2) (eq. 11.4) at the tuned parameter, and the
    tuned parameter itself, minimizing over a fine grid on the same range as ALPHA_GRID / MU_GRID.
    Overlaps: naive -> sts.naive_theory, A -> sts.methodA_theory, B -> sts.overlap_B_theory (uses n)."""
    grid = (np.linspace(ALPHA_GRID[0], ALPHA_GRID[-1], 201) if method == "naive"
            else np.geomspace(MU_GRID[0], MU_GRID[-1], 201))
    L_best, p_best = [], []
    for v in values:
        par = dict(lam1=lam1, lam2=lam2, rho=rho)
        par[vary] = v
        l1, l2, r = par["lam1"], par["lam2"], par["rho"]
        if method == "naive":
            ov = [sts.naive_theory(p, l1, l2, r)[:2] for p in grid]
        elif method == "A":
            ov = [sts.methodA_theory(p, l1, l2, r)[:2] for p in grid]
        else:
            ov = [sts.overlap_B_theory(l1, l2, r, p, n)[:2] for p in grid]
        L = np.array([-2 * (l1 * a ** 2 + l2 * b ** 2) for a, b in ov])
        k = np.argmin(L)
        L_best.append(L[k])
        p_best.append(grid[k])
    return np.array(L_best), np.array(p_best)


def compare_tuned(n=400, lam1=3.0, lam2=2.0, rho=0.2, M=10, mode="report",
                  ranges=None):
    """Naive, A and B, each with its parameter tuned by minimizing the simulated UNBIASED loss
    (fresh noise, eq. 11.3): alpha* on ALPHA_GRID for naive, mu_R* on MU_GRID for A and B. Numerics only.
    Columns: vs lam1, vs rho, vs lam2 (the other two fixed to lam1, lam2, rho).
    Top row   : unbiased loss - const at the tuned parameter, mean +- std/sqrt(M).
    Bottom row: tuned parameter, mu_R* (A, B, left log axis) and alpha* (naive, right axis)."""
    ss.set_style(mode)
    ranges = ranges or {"lam1": np.linspace(0.5, 4, 9), "rho": np.linspace(0.0, 0.95, 9),
                        "lam2": np.linspace(0.5, 4, 9)}
    for vary, values in ranges.items():
        _compare_tuned_one(vary, values, n, lam1, lam2, rho, M, mode)


def _compare_tuned_one(vary, values, n, lam1, lam2, rho, M, mode):
    """One figure, two panels side by side, for one varied parameter:
    (a) unbiased loss - const at the tuned parameter (points: simulation, lines: theory), plus the reference:
        -2 theta_+        (oracle without noise: max over ||x|| = 1 of lam1 m1^2 + lam2 m2^2 = top eigenvalue of
                           lam1 x1 x1^T + lam2 x2 x2^T, eq. 5.2 with a = lam1, b = lam2, dashed grey)
    (b) tuned parameters: mu_R* (A, B, left log axis), alpha* (naive, right axis)."""
    labels = {"lam1": r"$\lambda_1$", "rho": r"$\rho$", "lam2": r"$\lambda_2$"}
    res = _tuned_results(vary, values, n, lam1, lam2, rho, M)
    fig, (ax, ax_p) = ss.figure(ncols=2, aspect=1.0)
    ax_a = ax_p.twinx()
    ax_a.spines["right"].set_visible(True)

    # reference: best achievable without noise
    v_th = np.linspace(values[0], values[-1], 200)
    pars = [dict(dict(lam1=lam1, lam2=lam2, rho=rho), **{vary: v}) for v in v_th]
    ax.plot(v_th, [-2 * sts.eigval_P(p["lam1"], p["lam2"], p["rho"])[0] for p in pars], color="#888888", ls="--")

    print(f"vs {vary}:")
    for m, r in res.items():
        st = ss.METHOD[m]
        grid = ALPHA_GRID if m == "naive" else MU_GRID
        L = r[:, 1]                                          # unbiased, (len(values), M, len(grid))
        mean, err = L.mean(1), L.std(1) / np.sqrt(M)
        best = np.argmin(mean, axis=1)
        idx = np.arange(len(values))
        ax.errorbar(values, mean[idx, best], err[idx, best], color=st["color"], marker=st["marker"], ls="")
        ax_t = ax_a if m == "naive" else ax_p
        ax_t.plot(values, grid[best], color=st["color"], marker=st["marker"], ls="")
        # theory: L_eval = -2 (lam1 m1^2 + lam2 m2^2), minimized on a fine grid over the same range
        v_th = np.linspace(values[0], values[-1], 60)
        L_th, p_th = _tuned_theory(m, vary, v_th, n, lam1, lam2, rho)
        ax.plot(v_th, L_th, color=st["color"], ls="-")
        ax_t.plot(v_th, p_th, color=st["color"], ls=":" if m == "naive" else "-")
        print(f"  {m:5s} tuned param = {np.round(grid[best], 3)}")
    for a in (ax, ax_p):
        a.set_xlabel(labels[vary])
    ax_p.set_yscale("log")
    ax_a.set_ylim(0, 1.02)
    ax_a.grid(False)
    ax.set_ylabel(r"$\mathcal{L}_{\mathrm{eval}} - \mathrm{const}$ (tuned)")
    ax_p.set_ylabel(r"$\mu_R^*$ (A, B)")
    ax_a.set_ylabel(r"$\alpha^*$ (naive)")
    fixed = ", ".join(f"{labels[k][:-1]} = {v}$" for k, v in
                      dict(lam1=lam1, rho=rho, lam2=lam2).items() if k != vary)
    fig.suptitle(fixed)
    handles = [plt.Line2D([], [], color=ss.METHOD[m]["color"], marker=ss.METHOD[m]["marker"], ls="-",
                          label=ss.METHOD[m]["label"]) for m in ("naive", "A", "B")]
    handles += [plt.Line2D([], [], color=ss.INK, marker="o", ls="", label=f"simulation ($n={n}$, {M} samples)"),
                plt.Line2D([], [], color=ss.INK, ls="-", label=r"theory ($\mu_R^*$: solid, $\alpha^*$: dotted)"),
                plt.Line2D([], [], color="#888888", ls="--", label=r"$-2\theta_+$")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    ss.panel_labels([ax, ax_p])
    tag = "_".join(f"{k}={v}" for k, v in dict(lam1=lam1, lam2=lam2, rho=rho).items() if k != vary)
    ss.save(fig, f"loss_unbiased_tuned_vs_{vary}_{tag}_n={n}_{mode}", DIR_PLOTS)


# ---------------------------------------------------------------- where does method A beat the naive method? (theory)
def heatmap_gap_A(rhos=(0.1, 0.2, 0.3), lam_range=(0.5, 5.0), npts=91, vclip=0.4, mode="report"):
    """Theory (n -> inf): Delta = L_naive* - L_A* in the (lam1, lam2) plane, one panel per rho.
    Each method is tuned on the unbiased loss L_eval = -2 (lam1 m1^2 + lam2 m2^2):
    alpha* in [0, 1] (201 points), mu_R* in [1e-2, 1e3] (301 points, log).
    Delta > 0 (blue): method A better, Delta < 0 (red): naive better.
    Lines: Delta = +-0.01 (solid / dashed black), BBP thresholds lam1 = 1, lam2 = 1 (grey dotted)."""
    from matplotlib.colors import TwoSlopeNorm
    ss.set_style(mode)
    lams = np.linspace(*lam_range, npts)
    D = np.zeros((len(rhos), npts, npts))                    # D[k, i, j]: rho_k, lam2_i, lam1_j
    for k, rho in enumerate(rhos):
        for i, l2 in enumerate(lams):
            for j, l1 in enumerate(lams):
                D[k, i, j] = (sts.loss_tuned_naive_theory(l1, l2, rho)[0]
                              - sts.loss_tuned_A_theory(l1, l2, rho)[0])
        print(f"rho = {rho}: Delta in [{D[k].min():.3f}, {D[k].max():.3f}]", flush=True)

    norm = TwoSlopeNorm(vmin=-vclip, vcenter=0.0, vmax=vclip)    # clipped: the naive peak near lam1 = 1 is ~ -1.4
    fig, axes = ss.figure(ncols=len(rhos), aspect=1.0, sharex=True, sharey=True)
    for ax, rho, Dk in zip(axes, rhos, D):
        im = ax.pcolormesh(lams, lams, Dk, cmap="RdBu", norm=norm, shading="auto", rasterized=True)
        ax.contour(lams, lams, Dk, levels=[-0.01], colors=ss.INK, linestyles="--", linewidths=0.8)
        ax.contour(lams, lams, Dk, levels=[0.01], colors=ss.INK, linestyles="-", linewidths=0.8)
        ax.axvline(1, color="#888888", ls=":", lw=0.8)
        ax.axhline(1, color="#888888", ls=":", lw=0.8)
        ax.set_title(rf"$\rho = {rho}$")
        ax.set_aspect("equal")
        ax.grid(False)
    for ax in axes:
        ax.set_xlabel(r"$\lambda_1$")
    axes[0].set_ylabel(r"$\lambda_2$")
    fig.colorbar(im, ax=list(axes), shrink=0.8, extend="both",
                 label=r"$\Delta = \mathcal{L}^*_{\mathrm{naive}} - \mathcal{L}^*_{A}$")
    handles = [plt.Line2D([], [], color=ss.INK, ls="-", lw=0.8, label=r"$\Delta = +0.01$ (A better)"),
               plt.Line2D([], [], color=ss.INK, ls="--", lw=0.8, label=r"$\Delta = -0.01$ (naive better)"),
               plt.Line2D([], [], color="#888888", ls=":", lw=0.8, label=r"BBP threshold $\lambda_1 = 1$, $\lambda_2 = 1$")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    ss.panel_labels(axes)
    tag = "_".join(f"{r}" for r in rhos)
    ss.save(fig, f"heatmap_gap_naive_A_rho={tag}_{mode}", DIR_PLOTS)


if __name__ == "__main__":
    # python plot_maker_celia.py loss_mu_AB            -> report style
    # python plot_maker_celia.py loss_mu_AB slides     -> presentation style
    args = sys.argv[1:]
    mode = "presentation" if "slides" in args else "report"
    todo = [a for a in args if a != "slides"] or ["loss_mu"]
    if "loss_mu" in todo: loss_eval_vs_mu(mode=mode)
    if "loss_mu_AB" in todo: loss_eval_vs_mu(methods=("A", "B"), mode=mode)
    if "loss_mu_raw" in todo: loss_eval_vs_mu(methods=("A", "B"), mode=mode, raw=True)
    if "opt_A" in todo: optimality_A(mode=mode)              # weights lam_l   (Frobenius expansion)
    if "opt_A_p2" in todo: optimality_A(mode=mode, p=2)      # weights lam_l^2 (eq. 11.2 of the report)
    if "bias_naive" in todo: loss_naive_vs_alpha(mode=mode)  # biased / unbiased loss vs alpha (naive)
    if "bias_A" in todo: loss_fisher_vs_mu("A", mode=mode)   # biased / unbiased loss vs mu_R (method A)
    if "bias_B" in todo: loss_fisher_vs_mu("B", mode=mode)   # biased / unbiased loss vs mu_R (method B)
    if "tuned" in todo: compare_tuned(mode=mode)            # naive / A / B tuned on the unbiased loss
    if "gap_A" in todo:                                      # where A beats naive (theory heatmaps)
        heatmap_gap_A(rhos=(0.1, 0.2, 0.3), mode=mode)
        heatmap_gap_A(rhos=(0.5, 0.7, 0.9), mode=mode)
