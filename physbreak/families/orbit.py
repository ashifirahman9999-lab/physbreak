"""Orbit family: orbital periods from numerical integration of the equations of motion."""
import numpy as np
from scipy.integrate import solve_ivp

from .common import make_problem, r

FAMILY = "orbit"
GRAV = 6.674e-11


def _circular_period(rad, accel_mag):
    v = np.sqrt(accel_mag(rad) * rad)
    t_guess = 2 * np.pi * rad / v
    def rhs(t, s):
        x, y, vx, vy = s
        rr = np.hypot(x, y)
        a = accel_mag(rr)
        return [vx, vy, -a * x / rr, -a * y / rr]

    def done(t, s):
        return s[1]
    done.direction = 1  # y crosses 0 upward: completed a full turn (starts at y=0 moving up)

    sol = solve_ivp(rhs, [0, 3 * t_guess], [rad, 0.0, 0.0, v], events=done,
                    rtol=1e-11, atol=1e-11 * rad, max_step=t_guess / 500,
                    first_step=t_guess / 1e5)
    ts = [t for t in sol.t_events[0] if t > 0.25 * t_guess]
    return ts[0]


def _eccentric_period(rp, vp, mu):
    """Periapsis-to-periapsis time for a bound eccentric orbit."""
    a = 1.0 / (2.0 / rp - vp**2 / mu)
    t_kepler = 2 * np.pi * np.sqrt(a**3 / mu)

    def rhs(t, s):
        x, y, vx, vy = s
        rr = np.hypot(x, y)
        return [vx, vy, -mu * x / rr**3, -mu * y / rr**3]

    def peri(t, s):
        return s[0] * s[2] + s[1] * s[3]
    peri.terminal = True
    peri.direction = 1  # radial velocity - to +: passing periapsis

    sol = solve_ivp(rhs, [t_kepler * 0.5, t_kepler * 1.5],
                    _state_at(rp, vp, mu, t_kepler * 0.5), events=peri,
                    rtol=1e-12, atol=1e-12 * rp, method="DOP853")
    return sol.t_events[0][0]


def _state_at(rp, vp, mu, t):
    def rhs(tt, s):
        x, y, vx, vy = s
        rr = np.hypot(x, y)
        return [vx, vy, -mu * x / rr**3, -mu * y / rr**3]
    sol = solve_ivp(rhs, [0, t], [rp, 0.0, 0.0, vp], rtol=1e-12, atol=1e-12 * rp,
                    method="DOP853")
    return sol.y[:, -1]


def generate(level, n, rng):
    out = []
    for i in range(n):
        if level == 0:
            M = r(rng.uniform(1e24, 2e25))
            alt_km = r(rng.uniform(7000, 40000))
            mu = GRAV * M
            rad = alt_km * 1e3
            truth = _circular_period(rad, lambda x: mu / x**2) / 3600
            naive = 2 * np.pi * np.sqrt(rad**3 / mu) / 3600
            prompt = (f"A satellite moves in a circular orbit of radius {alt_km:g} km around a planet "
                      f"of mass {M:.3g} kg. Use G = 6.674e-11 N m^2 kg^-2. What is its orbital "
                      f"period in hours?")
            out.append(make_problem(FAMILY, level, i, prompt, truth, "hours",
                                    {"M": M, "r_km": alt_km}, naive))
        elif level == 1:
            mu = r(rng.uniform(1.0, 50.0))
            rad = r(rng.uniform(1.0, 8.0))
            truth = _circular_period(rad, lambda x: mu / x**2)
            naive = 2 * np.pi * np.sqrt(rad**3 / mu)
            prompt = (f"A pebble drifts around a heavy core in empty space. The core pulls on it with "
                      f"an acceleration of {mu} / d^2 (in reaches per beat squared), where d is the "
                      f"distance from the core in reaches. The pebble keeps a fixed distance of "
                      f"{rad} reaches from the core the whole time. How many beats does one full "
                      f"lap take?")
            out.append(make_problem(FAMILY, level, i, prompt, truth, "beats",
                                    {"mu": mu, "d": rad}, naive))
        elif level == 2:
            M = r(rng.uniform(1e24, 2e25))
            rp_km = r(rng.uniform(7000, 20000))
            frac = r(rng.uniform(0.85, 0.97))
            mu = GRAV * M
            rp = rp_km * 1e3
            vp = r(frac * np.sqrt(2 * mu / rp))
            truth = _eccentric_period(rp, vp, mu) / 3600
            naive = 2 * np.pi * np.sqrt(rp**3 / mu) / 3600  # treating r_p as a circular radius
            prompt = (f"A probe orbits a planet of mass {M:.3g} kg. At its closest approach it is "
                      f"{rp_km:g} km from the planet's centre, moving at {vp:g} m/s perpendicular to "
                      f"the line to the centre. Use G = 6.674e-11 N m^2 kg^-2. How long, in hours, "
                      f"until it is next at closest approach?")
            out.append(make_problem(FAMILY, level, i, prompt, truth, "hours",
                                    {"M": M, "rp_km": rp_km, "vp": vp}, naive))
        elif level == 3:
            p = float(rng.choice([1.5, 2.5, 3.0]))
            k = r(rng.uniform(1.0, 50.0))
            rad = r(rng.uniform(1.0, 8.0))
            truth = _circular_period(rad, lambda x: k / x**p)
            naive = 2 * np.pi * np.sqrt(rad**3 / k)  # Kepler's third law, wrongly reused
            prompt = (f"In a hypothetical universe, gravity produces an attractive acceleration of "
                      f"magnitude k / r^{p:g}, with k = {k} in SI-consistent units (r in metres, "
                      f"acceleration in m/s^2). A small moon moves in a perfectly circular orbit of "
                      f"radius {rad} m around a fixed central body. What is its orbital period in "
                      f"seconds?")
            out.append(make_problem(FAMILY, level, i, prompt, truth, "s",
                                    {"p": p, "k": k, "r": rad}, naive))
    return out
