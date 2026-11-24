import numpy as np
#from scipy.optimize import brant


def eigval_P(a,b,rho):
    """Compute the theoretical eigenvalues of the matrix P."""
    thetap = (a+b)/2 + np.sqrt((a-b)**2/4 + a*b*rho**2)
    thetam = (a+b)/2 - np.sqrt((a-b)**2/4 + a*b*rho**2)
    # lam = (a+b)/2 +- sqrt((a-b)^2 /4 + ab rho^2)
    return thetap, thetam

def eigvect_P(a,b,x1,x2,rho):

    thetap, thetam = eigval_P(a,b,rho)

    rp = (thetap-a)/(a*rho)
    vp = x1 + rp * x2
    vp /= np.linalg.norm(vp)

    rm = (thetam-a)/(a*rho)
    vm = x1 + rm * x2
    vm /= np.linalg.norm(vm)

    return vp, vm

def eigval_Y(a,b,rho, c=0):
    """ Top eig of Y from BBP """
    th= eigval_P(a,b,rho)
    return [theta + 1/np.maximum(theta, 1e-12) +c if theta > 1 else 2
        for theta in th]

def overlap_Y_P(theta):
    theta = np.asarray(theta)
    return np.where(
        theta > 1 +1e-12,
        np.sqrt(1 - 1 / theta**2),
        0
    )

def overlap_naive_signal(theta, rho, a, b):
    r = rho*a /(theta - a)
    N = np.sqrt(r**2 + 2*rho *r + 1)
    c1 = r/N
    ov = overlap_Y_P(theta)
    o1 = ov*theta*abs(c1)/a
    o2 = ov*theta*abs(2)/b
    return o1, o2

def beta_value(mu, lambda1, m1, lambda2, rho):  # builds the beta matrix, diagonalizes it, returns the top eigenvalue
    """Top eigenvalue of the 3x3 matrix beta.

    beta is the low-rank part of A = Y2 - I + 2 mu F1(x),  x = m1 x1 + sqrt(1 - m1^2) g_perp,
    i.e.  lambda2 x2 x2^T + 2 mu (lambda1 - 1)^2 x x^T  (1/N terms of F1 dropped),
    written in the orthonormal basis ( x1, (x2 - rho x1)/sqrt(1 - rho^2), g_perp ):

        beta11 = 2 mu (lambda1-1)^2 m1^2 + lambda2 rho^2
        beta22 = (1 - rho^2) lambda2
        beta33 = 2 mu (lambda1-1)^2 (1 - m1^2)
        beta12 = beta21 = lambda2 rho sqrt(1 - rho^2)
        beta13 = beta31 = 2 mu (lambda1-1)^2 m1 sqrt(1 - m1^2)
        beta23 = beta32 = 0

    Note: the top eigenvalue of A itself is then (BBP) theta + 1/theta - 1 if theta > 1,
    else the bulk edge 2 - 1 = 1, with theta = beta_value(...).
    """
    c = 2 * mu * (lambda1 - 1) ** 2
    s_rho = np.sqrt(max(0.0, 1 - rho ** 2))
    s_m = np.sqrt(max(0.0, 1 - m1 ** 2))
    beta = np.array([
        [c * m1 ** 2 + lambda2 * rho ** 2, lambda2 * rho * s_rho,    c * m1 * s_m],
        [lambda2 * rho * s_rho,            lambda2 * (1 - rho ** 2), 0.0],
        [c * m1 * s_m,                     0.0,                      c * (1 - m1 ** 2)],
    ])
    return np.linalg.eigvalsh(beta)[-1]

def overlap(x1, x2):
    return abs(np.dot(x1, x2))

def m1_theory(lam1):
    """BBP overlap between x1 and the top eigenvector of Y1: sqrt(1 - 1/lam1^2) if lam1 > 1, else 0."""
    if lam1 > 1:
        return np.sqrt(1 - 1 / lam1 ** 2)
    else:
        return 0.0


def semi_circle(z): #takes an array of values, return the semi circle for those values
    z = np.asarray(z)
    result = np.zeros_like(z, dtype=float)

    mask = np.abs(z) < 2 - 1e-12
    result[mask] = np.sqrt(4 - z[mask]**2) / (2 * np.pi) 

    return result


def overlap_A_theory(lam1, lam2, rho, mu):
    """Theoretical |<x1, v>|, v = top eigenvector of A, for x = x_hat or the BBP surrogate.
    beta is the same 3x3 matrix as in source_twospike.beta_value, in the basis
    (x1, (x2 - rho x1)/sqrt(1-rho^2), g_perp). With theta, u its top eigenvalue / eigenvector,
    BBP gives |<x1, v>| = sqrt(1 - 1/theta^2) * |u[0]| if theta > 1, else 0."""
    m1 = m1_theory(lam1)
    c = 2 * mu * (lam1 - 1) ** 2
    s_rho = np.sqrt(max(0.0, 1 - rho ** 2))
    s_m = np.sqrt(max(0.0, 1 - m1 ** 2))
    beta = np.array([
        [c * m1 ** 2 + lam2 * rho ** 2, lam2 * rho * s_rho,    c * m1 * s_m],
        [lam2 * rho * s_rho,            lam2 * (1 - rho ** 2), 0.0],
        [c * m1 * s_m,                  0.0,                   c * (1 - m1 ** 2)],
    ])
    w, U = np.linalg.eigh(beta)
    theta = w[-1]
    u = U[:, -1]
    if theta > 1:
        return np.sqrt(1 - 1 / theta ** 2) * abs(u[0])
    else:
        return 0.0


def top_eig_A_theory(lam1, lam2, rho, mu):
    """Theoretical top eigenvalue of A = Y2 - I + 2 mu F1(x), for x = x_hat, x_pg1 or x_pg12.
    theta = top eigenvalue of beta (beta_value). BBP: theta + 1/theta - 1 if theta > 1,
    else the bulk edge 2 - 1 = 1."""
    m1 = m1_theory(lam1)
    theta = beta_value(mu, lam1, m1, lam2, rho)
    if theta > 1:
        return theta + 1 / theta - 1
    else:
        return 1.0


# ---------------------------------------------------------------- phase diagram of A (report, section 8.4)
# A = Lam x_hat x_hat^T + lam2 x2 x2^T + Z2 + c I, with <x_hat, x2> -> m1 rho.
# Its signal eigenvalues are theta_{1,2} = eigval_P(Lam, lam2, m1 rho) (report Eq. 8.2),
# and theta_l gives an outlier iff theta_l > 1 (report Eq. 8.3).
# theta_l crosses 1 iff p(1) = (Lam - 1)(lam2 - 1) - Lam lam2 m1^2 rho^2 = 0, i.e.
#     lam2 = (Lam - 1) / (Lam (1 - m1^2 rho^2) - 1).
# Since p(Lam) <= 0 and p(lam2) <= 0, Lam and lam2 both lie between theta_2 and theta_1, so
#     Lam < 1 branch: 0 outlier below the curve, 1 above  -> f_01
#     Lam > 1 branch: 1 outlier below the curve, 2 above  -> f_12 (exists only if Lam (1 - m1^2 rho^2) > 1)

def Lambda_theory(lam1, mu, n):
    """Lam = 2 mu ((lam1 - 1)^2 + 1/n), weight of x_hat x_hat^T in A (report Eq. 8.4)."""
    return 2 * mu * ((lam1 - 1) ** 2 + 1 / n)


def f_01(lam1, rho, mu, n):
    """Boundary 0 outlier (below) / 1 outlier (above) of A in the (lam1, lam2) plane:
    lam2 = (1 - Lam) / (1 - Lam (1 - m1^2 rho^2)), defined where Lam < 1 (nan elsewhere)."""
    Lam = Lambda_theory(lam1, mu, n)
    m1 = m1_theory(lam1)
    if Lam < 1:
        return (1 - Lam) / (1 - Lam * (1 - m1 ** 2 * rho ** 2))
    else:
        return np.nan


def f_12(lam1, rho, mu, n):
    """Boundary 1 outlier (below) / 2 outliers (above) of A in the (lam1, lam2) plane:
    lam2 = (Lam - 1) / (Lam (1 - m1^2 rho^2) - 1), defined where Lam (1 - m1^2 rho^2) > 1 (nan elsewhere)."""
    Lam = Lambda_theory(lam1, mu, n)
    m1 = m1_theory(lam1)
    if Lam * (1 - m1 ** 2 * rho ** 2) > 1:
        return (Lam - 1) / (Lam * (1 - m1 ** 2 * rho ** 2) - 1)
    else:
        return np.nan


def n_outliers_theory(lam1, lam2, rho, mu, n):
    """Number of outliers of A (= number of theta_l > 1), theta_l from eigval_P with a = Lam, b = lam2, rho -> m1 rho."""
    Lam = Lambda_theory(lam1, mu, n)
    m1 = m1_theory(lam1)
    theta1, theta2 = eigval_P(Lam, lam2, m1 * rho)
    return int(theta1 > 1) + int(theta2 > 1)




def top_eig_A_theory_report(lam1, lam2, rho, mu, n):
    """Top eigenvalue of A, simplified version of the report (Eq. 9.2), with the 1/n terms kept.
    Lam = 2 mu ((lam1 - 1)^2 + 1/n) (Eq. 8.4), theta_max = theta_1 = eigval_P(Lam, lam2, m1 rho) (Eq. 8.2), c = 2 mu / n - 1:
        theta_max + 1/theta_max + 2 mu / n - 1   if theta_max > 1,
        1 + 2 mu / n                             else.
    The old version (top_eig_A_theory) is the same without the 1/n terms."""
    Lam = Lambda_theory(lam1, mu, n)
    m1 = m1_theory(lam1)
    theta_max, theta_2 = eigval_P(Lam, lam2, m1 * rho)
    if theta_max > 1:
        return theta_max + 1 / theta_max + 2 * mu / n - 1
    else:
        return 1 + 2 * mu / n


def rho_tilde_theory(rho, n):
    """Typical size of rho_tilde = <x2, x3>, x3 a random unit vector orthogonal to x1:
    x2 - rho x1 has norm sqrt(1 - rho^2) and spreads over the n - 1 directions orthogonal to x1,
    so rho_tilde^2 ~ (1 - rho^2) / (n - 1) ~ (1 - rho^2) / n."""
    return np.sqrt((1 - rho ** 2) / n)


def beta_value_finite_n(mu, lambda1, m1, lambda2, rho, n, rho_tilde):
    """Top eigenvalue of the 3x3 matrix beta of report Appendix B.2 (Table 1), finite n:
    same as beta_value, but with Lam = 2 mu ((lambda1 - 1)^2 + 1/n) and rho_tilde = <x2, x3> kept:

        beta11 = Lam m1^2 + lambda2 rho^2
        beta22 = lambda2 (1 - rho^2 - rho_tilde^2)
        beta33 = Lam (1 - m1^2) + lambda2 rho_tilde^2
        beta12 = beta21 = lambda2 rho sqrt(1 - rho^2 - rho_tilde^2)
        beta13 = beta31 = Lam m1 sqrt(1 - m1^2) + lambda2 rho rho_tilde
        beta23 = beta32 = lambda2 rho_tilde sqrt(1 - rho^2 - rho_tilde^2)
    """
    Lam = 2 * mu * ((lambda1 - 1) ** 2 + 1 / n)
    s_rho = np.sqrt(max(0.0, 1 - rho ** 2 - rho_tilde ** 2))
    s_m = np.sqrt(max(0.0, 1 - m1 ** 2))
    beta = np.array([
        [Lam * m1 ** 2 + lambda2 * rho ** 2, lambda2 * rho * s_rho,                       Lam * m1 * s_m + lambda2 * rho * rho_tilde],
        [lambda2 * rho * s_rho,              lambda2 * s_rho ** 2,                        lambda2 * rho_tilde * s_rho],
        [Lam * m1 * s_m + lambda2 * rho * rho_tilde, lambda2 * rho_tilde * s_rho,         Lam * (1 - m1 ** 2) + lambda2 * rho_tilde ** 2],
    ])
    return np.linalg.eigvalsh(beta)[-1]


def top_eig_A_theory_finite_n(lam1, lam2, rho, mu, n, rho_tilde=None):
    """Top eigenvalue of A with the full beta of Appendix B.2 (beta_value_finite_n) and c = 2 mu / n - 1 (as in Eq. 9.2):
    theta + 1/theta + 2 mu / n - 1 if theta > 1, else 1 + 2 mu / n.
    rho_tilde defaults to rho_tilde_theory(rho, n)."""
    if rho_tilde is None:
        rho_tilde = rho_tilde_theory(rho, n)
    m1 = m1_theory(lam1)
    theta = beta_value_finite_n(mu, lam1, m1, lam2, rho, n, rho_tilde)
    if theta > 1:
        return theta + 1 / theta + 2 * mu / n - 1
    else:
        return 1 + 2 * mu / n


def overlap_A_theory_report(lam1, lam2, rho, mu, n):
    """Overlap m1^A of Method A with x1 (report Eq. 9.3), x_hat = top eigenvector of Y1.
    Lam = 2 mu ((lam1 - 1)^2 + 1/n) (Eq. 8.4), theta_max = theta_1 = eigval_P(Lam, lam2, m1 rho) (Eq. 8.2),
    (c1, c2) = (r, 1) / sqrt(r^2 + 2 m1 rho r + 1), r = m1 rho Lam / (theta_max - Lam) (Eq. 8.7):
        m1^A = |m1 c1 + rho c2| sqrt(1 - theta_max^-2)
    for theta_max > 1, and 0 otherwise (overlap_Y_P)."""
    Lam = Lambda_theory(lam1, mu, n)
    m1 = m1_theory(lam1)
    theta_max, theta_2 = eigval_P(Lam, lam2, m1 * rho)
    if abs(theta_max - Lam) < 1e-12:      # m1 rho = 0 and Lam >= lam2: the top eigenvector of P is x_hat
        c1 = 1.0
        c2 = 0.0
    else:
        r = m1 * rho * Lam / (theta_max - Lam)
        N = np.sqrt(r ** 2 + 2 * m1 * rho * r + 1)
        c1 = r / N
        c2 = 1 / N
    ov = float(overlap_Y_P(theta_max))    # sqrt(1 - 1/theta_max^2) if theta_max > 1, else 0
    mA1 = abs(m1 * c1 + rho * c2) * ov
    return mA1


def top_eig_B(lam1, lam2, rho, rho_t, mu, n, k=-1):
    """Eigenvalue of the full 3x3 B of report Table 1 (k=-1: top theta_1, k=-2: second theta_2), i.e. with beta23 = beta32 != 0 when rho_t = <x2, x3> != 0.
    B = Lam v v^T + lam2 w w^T in the orthonormal basis e1 = x1, e3 = x3, e2 = (x2 - rho e1 - rho_t e3) / sqrt(1 - rho^2 - rho_t^2),
    with v = x_hat = (m1, 0, sqrt(1 - m1^2)) and w = x2 = (rho, sqrt(1 - rho^2 - rho_t^2), rho_t)."""
    Lam = Lambda_theory(lam1, mu, n)
    m1 = m1_theory(lam1)
    s_m = np.sqrt(1 - m1 ** 2)
    s_r = np.sqrt(max(0.0, 1 - rho ** 2 - rho_t ** 2))
    B = np.array([
        [Lam * m1 ** 2 + lam2 * rho ** 2,        lam2 * rho * s_r,    Lam * m1 * s_m + lam2 * rho * rho_t],
        [lam2 * rho * s_r,                       lam2 * s_r ** 2,     lam2 * rho_t * s_r],
        [Lam * m1 * s_m + lam2 * rho * rho_t,    lam2 * rho_t * s_r,  Lam * (1 - m1 ** 2) + lam2 * rho_t ** 2],
    ])
    return np.linalg.eigvalsh(B)[k]
