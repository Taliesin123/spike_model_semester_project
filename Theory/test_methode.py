"""Test of the theoretical overlaps of Method B (report Section 10), against simulations.

Theory (derivation: resolvent of A + Woodbury, see chat 24/11):
    A = c I + W + sum_l theta_l v_l v_l^T,   c = 2 mu / n - 1,
    theta_l, v_l = eigenvalues / eigenvectors of the 3x3 matrix B (top_eig_B with rho_t = 0),
    x_B = kappa R_A(lam) x_hat,   kappa = Lam + 2 mu / n   (b = 2 mu F1(x_hat) x_hat = kappa x_hat),
    z = lam - c,  g(z) = (-z + sqrt(z^2 - 4)) / 2,  g'(z) = g^2 / (1 - g^2),
    s_l = <v_l, x_hat>,  q_il = <v_l, x_i>:

    overlaps:   m_i^B = | kappa sum_l q_il s_l g / (1 + theta_l g) |
    norm:       kappa^2 sum_l s_l^2 g' / (1 + theta_l g)^2 = 1

Nothing in the existing files is modified: this file only imports them.
Usage:  python test_methode.py check heat_B1 heat_B2
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

import source_twospike as sts
import class_twospike
from class_twospike import TwoSpikes, solve_norm_constrained

# solve_norm_constrained calls brentq, but class_twospike.py never imports it (NameError).
# We do not touch class_twospike.py: we just give the module brentq from here.
class_twospike.brentq = brentq

ALPHA = np.sqrt(0.5)
DIR_PLOTS = "results"
os.makedirs(DIR_PLOTS, exist_ok=True)


# ---------------------------------------------------------------- theory
def g_semicircle(z):
    """Stieltjes transform of the semicircle on [-2, 2], for real z > 2. Value in (-1, 0)."""
    return (-z + np.sqrt(z ** 2 - 4)) / 2


def dg_semicircle(z):
    """Derivative g'(z) = g^2 / (1 - g^2) (from g^2 + z g + 1 = 0)."""
    g = g_semicircle(z)
    return g ** 2 / (1 - g ** 2)


def B_eig(lam1, lam2, rho, mu, n):
    """Eigenvalues theta (ascending) and eigenvectors V (columns) of the 3x3 matrix B,
    same B as sts.top_eig_B with rho_t = 0, in the basis e1 = x1, e2 = (x2 - rho x1)/sqrt(1 - rho^2), e3 = x3."""
    Lam = sts.Lambda_theory(lam1, mu, n)
    m1 = sts.m1_theory(lam1)
    s_m = np.sqrt(1 - m1 ** 2)
    s_r = np.sqrt(max(0.0, 1 - rho ** 2))
    B = np.array([
        [Lam * m1 ** 2 + lam2 * rho ** 2, lam2 * rho * s_r, Lam * m1 * s_m],
        [lam2 * rho * s_r,                lam2 * s_r ** 2,  0.0],
        [Lam * m1 * s_m,                  0.0,              Lam * (1 - m1 ** 2)],
    ])
    theta, V = np.linalg.eigh(B)
    return theta, V


def overlap_B_theory(lam1, lam2, rho, mu, n):
    """Theoretical m1^B = |<x_B, x1>|, m2^B = |<x_B, x2>| and Lagrange multiplier lam of Method B."""
    Lam = sts.Lambda_theory(lam1, mu, n)
    m1 = sts.m1_theory(lam1)
    c = 2 * mu / n - 1
    kappa = Lam + 2 * mu / n

    theta, V = B_eig(lam1, lam2, rho, mu, n)

    # x_hat, x1, x2 in the basis (e1, e2, e3), then in the eigenbasis of B
    x_hat = np.array([m1, 0.0, np.sqrt(1 - m1 ** 2)])
    x1 = np.array([1.0, 0.0, 0.0])
    x2 = np.array([rho, np.sqrt(max(0.0, 1 - rho ** 2)), 0.0])
    s = V.T @ x_hat
    q1 = V.T @ x1
    q2 = V.T @ x2

    # right end of the spectrum of A - c I: bulk edge 2, or the top outlier theta + 1/theta
    z_star = 2.0
    for l in range(3):
        if theta[l] > 1:
            z_star = max(z_star, theta[l] + 1 / theta[l])

    def Phi(z):
        """||x_B||^2 / kappa^2. Only the l with s_l != 0 enter (the others have no pole)."""
        g = g_semicircle(z)
        dg = dg_semicircle(z)
        total = 0.0
        for l in range(3):
            if abs(s[l]) > 1e-12:
                total += s[l] ** 2 * dg / (1 + theta[l] * g) ** 2
        return total

    def regular_overlap(z, q):
        """kappa sum_l q_l s_l g / (1 + theta_l g), over the l with s_l != 0."""
        g = g_semicircle(z)
        total = 0.0
        for l in range(3):
            if abs(s[l]) > 1e-12:
                total += q[l] * s[l] * g / (1 + theta[l] * g)
        return kappa * total

    z_lo = z_star + 1e-10
    if kappa ** 2 * Phi(z_lo) > 1:
        # standard case: unique root of kappa^2 Phi(z) = 1 on (z_star, inf)
        z_hi = z_star + 1.0
        while kappa ** 2 * Phi(z_hi) > 1:
            z_hi += 1.0
        z = brentq(lambda zz: kappa ** 2 * Phi(zz) - 1, z_lo, z_hi)
        mB1 = abs(regular_overlap(z, q1))
        mB2 = abs(regular_overlap(z, q2))
    else:
        # hard case: x_hat has no component on the top outlier (e.g. lam1 <= 1, or rho = 0),
        # the norm cannot reach 1 above z_star: lam sticks to the top outlier, and the missing norm
        # tau^2 = 1 - kappa^2 Phi goes on the top eigenvector u_top, with <u_top, x_i> -> sqrt(1 - theta^-2) q_i,top.
        # In these cases the regular part and the u_top part never overlap the same x_i, so we add the absolute values.
        z = z_star
        tau = np.sqrt(max(0.0, 1 - kappa ** 2 * Phi(z)))
        top = np.argmax(theta)
        ov = float(sts.overlap_Y_P(theta[top]))      # sqrt(1 - 1/theta^2)
        mB1 = abs(regular_overlap(z, q1)) + tau * ov * abs(q1[top])
        mB2 = abs(regular_overlap(z, q2)) + tau * ov * abs(q2[top])

    lam = c + z
    return mB1, mB2, lam


# ---------------------------------------------------------------- simulation
def overlap_B_one_draw(n, lam1, lam2, rho, mu, alpha=ALPHA):
    """One draw. x_B from the existing solver solve_norm_constrained, with
    A = Y2 - I + 2 mu F1(x_hat) (S.A) and b = 2 mu F1(x_hat) x_hat, x_hat = top eigenvector of Y1.
    Returns |<x_B, x1>|, |<x_B, x2>|, lam."""
    S = TwoSpikes(n, lam1, lam2, rho, mu=mu, alpha=alpha)
    _, x_hat = S.top_eigenvector(S.Y1)
    A = S.A(x_hat, mu)
    b = 2 * mu * S.fisher_MS(x_hat, lam1) @ x_hat
    x_B, lam = solve_norm_constrained(A, b)
    return sts.overlap(S.get_x1(), x_B), sts.overlap(S.get_x2(), x_B), lam


def overlap_B_grid(lam1_values, rho_values, lam2, mu, n, reps, which=1, alpha=ALPHA):
    """Simulated m_which^B, mean over reps draws, shape (len(rho_values), len(lam1_values))."""
    Z = np.zeros((len(rho_values), len(lam1_values)))
    for i, rho in enumerate(rho_values):
        for j, lam1 in enumerate(lam1_values):
            list_o = []
            for _ in range(reps):
                o1, o2, _ = overlap_B_one_draw(n, lam1, lam2, rho, mu, alpha=alpha)
                if which == 1:
                    list_o.append(o1)
                else:
                    list_o.append(o2)
            Z[i, j] = np.mean(list_o)
    return Z


# ---------------------------------------------------------------- checks and plots
def check(n=2000, lam2=3.0, mu=0.5, reps=5):
    """Prints theory vs simulation at a few points."""
    points = [(2.0, 0.5), (3.0, 0.3), (1.5, 0.8), (0.7, 0.5), (2.5, 0.0)]
    print(f"n={n}, lam2={lam2}, mu={mu}, {reps} draws")
    print("lam1  rho  |  m1 th   m1 sim  |  m2 th   m2 sim  |  lam th   lam sim")
    for lam1, rho in points:
        mB1, mB2, lam = overlap_B_theory(lam1, lam2, rho, mu, n)
        sims = np.array([overlap_B_one_draw(n, lam1, lam2, rho, mu) for _ in range(reps)])
        o1, o2, l = sims.mean(axis=0)
        print(f"{lam1:4.1f} {rho:4.1f}  |  {mB1:.3f}   {o1:.3f}   |  {mB2:.3f}   {o2:.3f}   |  {lam:.3f}   {l:.3f}")


def heat_overlap_B(lam2=3.0, mu=0.5, n=300, reps=5, which=1, levels=(0.2, 0.4, 0.6, 0.8, 0.95)):
    """Same plot as heat_overlap_A (plot_maker_tali.py), for Method B.
    Grid: simulated m_which^B. Red: level lines of overlap_B_theory."""
    lam1_values = np.linspace(0.5, 4, 30)
    rho_values = np.linspace(0, 1, 30)
    lam1_fine = np.linspace(0.5, 4, 300)          # for the theory
    rho_fine = np.linspace(0, 1, 300)

    Z = overlap_B_grid(lam1_values, rho_values, lam2, mu, n, reps, which=which)

    theory = np.zeros((len(rho_fine), len(lam1_fine)))
    for i, rho in enumerate(rho_fine):
        for j, lam1 in enumerate(lam1_fine):
            mB1, mB2, _ = overlap_B_theory(lam1, lam2, rho, mu, n)
            if which == 1:
                theory[i, j] = mB1
            else:
                theory[i, j] = mB2

    plt.figure(figsize=(7, 5.5))
    plt.pcolormesh(lam1_values, rho_values, Z, cmap="Greys", vmin=0, vmax=1, shading="nearest")
    plt.colorbar(label=rf"$m_{which}^B$ (simulation)")
    cs = plt.contour(lam1_fine, rho_fine, theory, levels=list(levels), colors="red", linewidths=1.5)
    plt.clabel(cs, fontsize=8)
    plt.plot([], [], color="red", lw=1.5, label=rf"theory $m_{which}^B$ (resolvent)")
    plt.xlabel(r"$\lambda_1$")
    plt.ylabel(r"$\rho$")
    plt.title(rf"$m_{which}^B = |\langle \hat x_B^{{(2)}}, x_{which}\rangle|$" + "\n"
              + rf"$\lambda_2$={lam2}, $\mu$={mu}, $n$={n}, {reps} draws per point", fontsize=10)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS, f"heat_overlap_B{which}_lam2={lam2}_mu={mu}_n={n}.png"), dpi=120)
    # no plt.close(): the figure stays open and is shown by plt.show() at the end


def overlap_B_vs_lam1(n=300, lam2=3.0, mu=0.5, rho=0.5, sims=20):
    """Same plot as overlap_x1_vs_lam1 (plot_maker_tali.py), for Method B.
    Points: simulated m1^B and m2^B (mean +- standard error). Red: overlap_B_theory."""
    lam1_values = np.linspace(0, 4, 21)

    mean_1, err_1 = [], []
    mean_2, err_2 = [], []

    for lam1 in lam1_values:
        list_1, list_2 = [], []
        for s in range(sims):
            o1, o2, _ = overlap_B_one_draw(n, lam1, lam2, rho, mu)
            list_1.append(o1)
            list_2.append(o2)
        mean_1.append(np.mean(list_1))
        err_1.append(np.std(list_1) / np.sqrt(sims))
        mean_2.append(np.mean(list_2))
        err_2.append(np.std(list_2) / np.sqrt(sims))

    lam1_fine = np.linspace(0, 4, 300)
    theory_1 = []
    theory_2 = []
    for lam1 in lam1_fine:
        mB1, mB2, _ = overlap_B_theory(lam1, lam2, rho, mu, n)
        theory_1.append(mB1)
        theory_2.append(mB2)

    plt.figure(figsize=(7, 4.8))
    plt.errorbar(lam1_values, mean_1, err_1, color="C0", marker="o", ms=5, ls="none", capsize=2, label=r"$m_1^B = |\langle x_1, \hat x_B\rangle|$ (simulation)")
    plt.errorbar(lam1_values, mean_2, err_2, color="C2", marker="s", ms=5, ls="none", capsize=2, label=r"$m_2^B = |\langle x_2, \hat x_B\rangle|$ (simulation)")
    plt.plot(lam1_fine, theory_1, color="red", ls="-", label=r"theory $m_1^B$")
    plt.plot(lam1_fine, theory_2, color="red", ls="--", label=r"theory $m_2^B$")
    plt.axvline(1, color="k", ls=":", label=r"BBP threshold $\lambda_1=1$")
    plt.xlabel(r"$\lambda_1$")
    plt.ylabel("overlap")
    plt.title(rf"Method B: $\rho$={rho}, $n$={n}, $\lambda_2$={lam2}, $\mu$={mu}, {sims} sims", fontsize=10)
    plt.grid()
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS, f"overlap_B_vs_lam1_rho={rho}_n={n}.png"), dpi=120)
    # no plt.close(): the figure stays open and is shown by plt.show() at the end


if __name__ == "__main__":
    todo = sys.argv[1:] or ["check", "heat_B1", "heat_B2", "B_lam1"]
    if "check" in todo: check()
    if "heat_B1" in todo: heat_overlap_B(which=1)
    if "heat_B2" in todo: heat_overlap_B(which=2)
    if "B_lam1" in todo: overlap_B_vs_lam1()
    plt.show()
