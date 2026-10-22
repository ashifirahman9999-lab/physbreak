"""Validate the numerical ground truth against exact results wherever they exist.

Run: python -m pytest tests  (or: python tests/test_ground_truth.py)
"""
import numpy as np
from scipy.integrate import quad
from scipy.special import ellipk

from physbreak.families import pendulum, projectile, orbit, cyclotron, quantum_well as qw


def close(a, b, tol=1e-4):
    assert abs(a - b) <= tol * abs(b), (a, b)


def test_pendulum_large_amplitude_exact():
    # exact period: T = 4 sqrt(L/g) K(sin(theta0/2)), scipy ellipk takes m = k^2
    L, g, a0 = 1.3, 9.81, np.radians(150)
    exact = 4 * np.sqrt(L / g) * ellipk(np.sin(a0 / 2) ** 2)
    close(pendulum._period(lambda th: -(g / L) * np.sin(th), a0), exact)


def test_pendulum_cubic_exact():
    # theta'' = -k theta^3: T = (4 / (A sqrt(k))) * integral_0^1 du / sqrt((1 - u^4)/2)
    k, A = 7.0, 0.8
    I = quad(lambda u: 1 / np.sqrt((1 - u**4) / 2), 0, 1)[0]
    close(pendulum._period(lambda th: -k * th**3, A), 4 * I / (A * np.sqrt(k)))


def test_projectile_vacuum_exact():
    v0, ang, g = 25.0, 40.0, 9.81
    close(projectile._range(v0, ang, g), v0**2 * np.sin(2 * np.radians(ang)) / g)


def test_orbit_eccentric_matches_kepler():
    mu, rp = 6.674e-11 * 6e24, 9e6
    vp = 0.93 * np.sqrt(2 * mu / rp)
    a = 1 / (2 / rp - vp**2 / mu)
    close(orbit._eccentric_period(rp, vp, mu), 2 * np.pi * np.sqrt(a**3 / mu))


def test_orbit_power_law_circular():
    k, rad, p = 12.0, 3.0, 2.5
    exact = 2 * np.pi * rad ** ((p + 1) / 2) / np.sqrt(k)
    close(orbit._circular_period(rad, lambda x: k / x**p), exact)


def test_cyclotron_relativistic():
    beta, B = 0.95, 0.1
    v = beta * cyclotron.C
    gamma = 1 / np.sqrt(1 - beta**2)
    exact = gamma * cyclotron.M_E * v / (cyclotron.E_CHARGE * B)
    close(cyclotron._radius(cyclotron.M_E, cyclotron.E_CHARGE, v, B, relativistic=True), exact)


def test_cyclotron_counterfactual():
    m, q, v, B, u = 2.0, 1.5, 6.0, 0.7, 4.0
    exact = m * u**2 / (q * B * v)  # m v^2 / r = q v B (v/u)^2
    close(cyclotron._radius(m, q, v, B, speed_factor=lambda s: (s / u) ** 2), exact)


def test_infinite_well():
    L = 0.8
    e = qw._converged(lambda x: np.zeros_like(x), L / 2, qw.HB2_2ME)
    close(e, qw.HB2_2ME * np.pi**2 / L**2, tol=1e-3)


def test_harmonic_oscillator_solver():
    # same solver on V = x^2/2 (hbar = m = 1) must give exactly 1/2
    close(qw._converged(lambda x: 0.5 * x**2, 8.0, 0.5), 0.5, tol=1e-4)


def test_cubic_potential_scaling():
    # exact scaling law for V = lambda |x|^3: E0(lambda) = E0(1) * lambda^(2/5)
    e1 = qw._converged(lambda x: np.abs(x) ** 3, 8.0, 0.5)
    e3 = qw._converged(lambda x: 3 * np.abs(x) ** 3, 8.0, 0.5)
    close(e3, e1 * 3 ** 0.4, tol=1e-3)


def test_finite_well_exact_vs_numerical():
    L, V0 = 0.5, 1.0
    exact = qw._finite_well_exact(L, V0, qw.HB2_2ME)
    fd = qw._ground_energy(lambda x: np.where(np.abs(x) <= L / 2, 0.0, V0),
                           np.linspace(-L / 2 - 4, L / 2 + 4, 40000)[1:-1], qw.HB2_2ME)
    close(fd, exact, tol=5e-3)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
