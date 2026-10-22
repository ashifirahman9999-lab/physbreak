"""Charged particle in a uniform magnetic field: orbit radius from numerical integration."""
import numpy as np
from scipy.integrate import solve_ivp

from .common import make_problem, r

FAMILY = "cyclotron"
C = 2.99792458e8
E_CHARGE = 1.602176634e-19
M_P = 1.67262192e-27
M_E = 9.1093837e-31


def _radius(m, q, v0, B, relativistic=False, speed_factor=None):
    """Orbit radius for motion in the x-y plane with B along z.

    The particle starts at the origin moving along +x, so the orbit centre lies on the y axis
    and the radius is half the maximum |y| reached.
    speed_factor(speed) multiplies the magnetic force (for counterfactual force laws).
    """
    gamma0 = 1 / np.sqrt(1 - (v0 / C) ** 2) if relativistic else 1.0
    p0 = gamma0 * m * v0
    omega_guess = q * B / (gamma0 * m)
    if speed_factor is not None:
        omega_guess *= speed_factor(v0)
    t_end = 1.2 * 2 * np.pi / omega_guess

    def rhs(t, s):
        x, y, px, py = s
        p = np.hypot(px, py)
        if relativistic:
            g = np.sqrt(1 + (p / (m * C)) ** 2)
        else:
            g = 1.0
        vx, vy = px / (g * m), py / (g * m)
        f = q * B
        if speed_factor is not None:
            f *= speed_factor(np.hypot(vx, vy))
        # F = q v x B with B = B z_hat: (vy*B, -vx*B)
        return [vx, vy, f * vy, -f * vx]

    r_scale = p0 / (m * omega_guess * gamma0)  # rough orbit size
    atol = [1e-10 * r_scale, 1e-10 * r_scale, 1e-10 * p0, 1e-10 * p0]
    sol = solve_ivp(rhs, [0, t_end], [0, 0, p0, 0], rtol=1e-11, atol=atol,
                    max_step=t_end / 4000, dense_output=True)
    ys = sol.sol(np.linspace(0, t_end, 20001))[1]
    return 0.5 * np.max(np.abs(ys))


def generate(level, n, rng):
    out = []
    for i in range(n):
        if level == 0:
            v = r(rng.uniform(1e5, 2e6))
            B = r(rng.uniform(0.1, 2.0))
            truth = _radius(M_P, E_CHARGE, v, B) * 1e3
            naive = M_P * v / (E_CHARGE * B) * 1e3
            prompt = (f"A proton moves at {v:.3g} m/s perpendicular to a uniform magnetic field of "
                      f"{B} T. Use proton mass 1.6726e-27 kg and charge 1.602e-19 C. What is the "
                      f"radius of its circular path in millimetres?")
            out.append(make_problem(FAMILY, level, i, prompt, truth, "mm",
                                    {"v": v, "B": B}, naive))
        elif level == 1:
            m = r(rng.uniform(0.5, 5.0))
            q = r(rng.uniform(0.5, 5.0))
            v = r(rng.uniform(1.0, 20.0))
            B = r(rng.uniform(0.2, 3.0))
            truth = _radius(m, q, v, B)
            naive = m * v / (q * B)
            prompt = (f"A bead of heft {m} grains carries {q} sparks of charge. It glides at {v} "
                      f"paces per tick through a region filled with a uniform 'swirl field' of "
                      f"strength {B} swirls, at right angles to the field. In these units the "
                      f"sideways push on the bead is (charge x speed x field strength), always at "
                      f"right angles to its motion, and push = heft x acceleration. The bead traces "
                      f"a circle. What is the radius of that circle in paces?")
            out.append(make_problem(FAMILY, level, i, prompt, truth, "paces",
                                    {"m": m, "q": q, "v": v, "B": B}, naive))
        elif level == 2:
            beta = r(rng.uniform(0.8, 0.99))
            B = r(rng.uniform(0.01, 0.5))
            v = beta * C
            truth = _radius(M_E, E_CHARGE, v, B, relativistic=True) * 1e3
            naive = M_E * v / (E_CHARGE * B) * 1e3
            prompt = (f"An electron moves at {beta}c perpendicular to a uniform magnetic field of "
                      f"{B} T. Use electron mass 9.109e-31 kg, charge 1.602e-19 C and "
                      f"c = 2.998e8 m/s. What is the radius of its circular path in millimetres?")
            out.append(make_problem(FAMILY, level, i, prompt, truth, "mm",
                                    {"beta": beta, "B": B}, naive))
        elif level == 3:
            m = r(rng.uniform(0.5, 5.0))
            q = r(rng.uniform(0.5, 5.0))
            v = r(rng.uniform(2.0, 20.0))
            B = r(rng.uniform(0.2, 3.0))
            u = r(rng.uniform(2.0, 20.0))
            truth = _radius(m, q, v, B, speed_factor=lambda s: (s / u) ** 2)
            naive = m * v / (q * B)
            prompt = (f"In a hypothetical universe, the magnetic force on a moving charge has the "
                      f"usual direction (perpendicular to both velocity and field) but magnitude "
                      f"q*v*B*(v/u)^2, where u = {u} m/s is a universal constant. A particle of mass "
                      f"{m} kg and charge {q} C moves at {v} m/s perpendicular to a uniform field "
                      f"of {B} T. There are no other forces. What is the radius of its circular "
                      f"path in metres?")
            out.append(make_problem(FAMILY, level, i, prompt, truth, "m",
                                    {"m": m, "q": q, "v": v, "B": B, "u": u}, naive))
    return out
