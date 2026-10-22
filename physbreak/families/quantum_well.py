"""Quantum well family: ground-state energies from a finite-difference Schrodinger solver."""
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq

from .common import make_problem, r

FAMILY = "quantum_well"
HB2_2ME = 0.0380998  # hbar^2 / (2 m_e) in eV nm^2


def _ground_energy(V, x, kin):
    """Lowest eigenvalue of -kin * psi'' + V(x) psi on grid x with psi = 0 at both ends.

    kin = hbar^2 / (2m) in the units of the problem.
    """
    h = x[1] - x[0]
    diag = 2 * kin / h**2 + V(x)
    off = -kin / h**2 * np.ones(len(x) - 1)
    w = eigh_tridiagonal(diag, off, select="i", select_range=(0, 0),
                         eigvals_only=True)
    return w[0]


def _converged(V, half_width, kin, n=4000):
    """Energy at two resolutions; returns the finer one (and checks they agree)."""
    e1 = _ground_energy(V, np.linspace(-half_width, half_width, n)[1:-1], kin)
    e2 = _ground_energy(V, np.linspace(-half_width, half_width, 2 * n)[1:-1], kin)
    assert abs(e1 - e2) <= 2e-3 * abs(e2) + 1e-12, (e1, e2)
    return e2


def _finite_well_exact(L, V0, kin):
    """Even ground state of a finite square well: solve k tan(kL/2) = kappa exactly."""
    def f(E):
        k = np.sqrt(E / kin)
        kap = np.sqrt((V0 - E) / kin)
        return k * np.tan(k * L / 2) - kap
    # ground state has k L / 2 < pi / 2
    e_max = min(V0, kin * (np.pi / L) ** 2) * (1 - 1e-12)
    return brentq(f, 1e-12 * V0, e_max, xtol=1e-14)


def generate(level, n, rng):
    out = []
    zero = lambda x: np.zeros_like(x)  # noqa: E731
    for i in range(n):
        if level == 0:
            L = r(rng.uniform(0.2, 2.0))
            truth = _converged(zero, L / 2, HB2_2ME)
            naive = HB2_2ME * np.pi**2 / L**2
            prompt = (f"An electron is confined in a one-dimensional infinite square well of width "
                      f"{L} nm. Use hbar^2/(2 m_e) = 0.0381 eV nm^2. What is its ground-state energy "
                      f"in eV?")
            out.append(make_problem(FAMILY, level, i, prompt, truth, "eV",
                                    {"L_nm": L}, naive))
        elif level == 1:
            m = r(rng.uniform(0.5, 5.0))
            L = r(rng.uniform(0.5, 5.0))
            kin = 1 / (2 * m)
            truth = _converged(zero, L / 2, kin)
            naive = np.pi**2 / (2 * m * L**2)
            prompt = (f"Work in units where the reduced Planck constant equals 1. A particle of mass "
                      f"{m} units can move freely along a line segment of length {L} units, but the "
                      f"walls at both ends are impenetrable. What is the lowest energy it can have, "
                      f"in these units?")
            out.append(make_problem(FAMILY, level, i, prompt, truth, "energy units",
                                    {"m": m, "L": L}, naive))
        elif level == 2:
            L = r(rng.uniform(0.3, 1.0))
            V0 = r(rng.uniform(0.3, 2.0))
            V = lambda x, L=L, V0=V0: np.where(np.abs(x) <= L / 2, 0.0, V0)  # noqa: E731
            truth = _finite_well_exact(L, V0, HB2_2ME)
            # cross-check against the finite-difference solver (1% agreement)
            fd = _ground_energy(V, np.linspace(-L / 2 - 4, L / 2 + 4, 20000)[1:-1], HB2_2ME)
            assert abs(fd - truth) < 0.01 * truth, (fd, truth)
            naive = HB2_2ME * np.pi**2 / L**2
            prompt = (f"An electron is in a one-dimensional finite square well of width {L} nm and "
                      f"depth {V0} eV (potential 0 inside, {V0} eV outside). Use "
                      f"hbar^2/(2 m_e) = 0.0381 eV nm^2. What is its ground-state energy in eV, "
                      f"measured from the bottom of the well?")
            out.append(make_problem(FAMILY, level, i, prompt, truth, "eV",
                                    {"L_nm": L, "V0_eV": V0}, naive))
        elif level == 3:
            lam = r(rng.uniform(0.5, 5.0))
            V = lambda x, lam=lam: lam * np.abs(x) ** 3  # noqa: E731
            truth = _converged(V, 8.0, 0.5)
            prompt = (f"Work in units where hbar = 1 and the particle mass m = 1. A particle moves in "
                      f"one dimension in the potential V(x) = {lam} * |x|^3. What is its ground-state "
                      f"energy?")
            out.append(make_problem(FAMILY, level, i, prompt, truth, "energy units",
                                    {"lambda": lam}, None))
    return out
