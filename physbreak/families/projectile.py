"""Projectile family: horizontal range on flat ground, from numerical integration."""
import numpy as np
from scipy.integrate import solve_ivp

from .common import make_problem, r

FAMILY = "projectile"
G = 9.81


def _range(v0, angle_deg, g, drag=None):
    """Horizontal distance travelled before returning to launch height.

    drag(speed) -> deceleration magnitude, directed opposite to the velocity.
    """
    th = np.radians(angle_deg)

    def rhs(t, s):
        x, y, vx, vy = s
        ax, ay = 0.0, -g
        if drag is not None:
            sp = np.hypot(vx, vy)
            if sp > 0:
                a = drag(sp)
                ax -= a * vx / sp
                ay -= a * vy / sp
        return [vx, vy, ax, ay]

    def land(t, s):
        return s[1]
    land.terminal = True
    land.direction = -1

    sol = solve_ivp(rhs, [0, 1e4], [0, 0, v0 * np.cos(th), v0 * np.sin(th)],
                    events=land, rtol=1e-10, atol=1e-10, max_step=0.01,
                    first_step=1e-6)
    return sol.y_events[0][0][0]


def generate(level, n, rng):
    out = []
    for i in range(n):
        if level == 0:
            v0 = r(rng.uniform(10, 40))
            ang = r(rng.uniform(20, 70), 2)
            truth = _range(v0, ang, G)
            naive = v0**2 * np.sin(2 * np.radians(ang)) / G
            prompt = (f"A ball is launched from flat ground at {v0} m/s at {ang} degrees above the "
                      f"horizontal. Neglect air resistance and take g = 9.81 m/s^2. How far away, "
                      f"in metres, does it land?")
            out.append(make_problem(FAMILY, level, i, prompt, truth, "m",
                                    {"v0": v0, "angle": ang}, naive))
        elif level == 1:
            v0 = r(rng.uniform(5, 30))
            ang = r(rng.uniform(20, 70), 2)
            gp = r(rng.uniform(1.5, 12))
            truth = _range(v0, ang, gp)
            naive = v0**2 * np.sin(2 * np.radians(ang)) / gp
            prompt = (f"On a flat plain where every loose object accelerates toward the ground at "
                      f"{gp} strides per beat squared, and there is no air at all, a pebble is "
                      f"flicked from the ground at {v0} strides per beat, aimed {ang} degrees up from "
                      f"the plain. How many strides from the flicking point does it come back down?")
            out.append(make_problem(FAMILY, level, i, prompt, truth, "strides",
                                    {"v0": v0, "angle": ang, "g": gp}, naive))
        elif level == 2:
            m = r(rng.uniform(0.05, 0.5))
            c = r(rng.uniform(0.002, 0.01))
            v0 = r(rng.uniform(40, 80))
            ang = r(rng.uniform(25, 60), 2)
            truth = _range(v0, ang, G, drag=lambda s: c * s**2 / m)
            naive = v0**2 * np.sin(2 * np.radians(ang)) / G
            prompt = (f"A ball of mass {m} kg is launched from flat ground at {v0} m/s at {ang} "
                      f"degrees above the horizontal. Air resistance is a force of magnitude "
                      f"c*v^2 opposite to the velocity, with c = {c} kg/m. Take g = 9.81 m/s^2. "
                      f"How far away, in metres, does it land?")
            out.append(make_problem(FAMILY, level, i, prompt, truth, "m",
                                    {"m": m, "c": c, "v0": v0, "angle": ang}, naive))
        elif level == 3:
            m = r(rng.uniform(0.1, 1.0))
            vt = rng.uniform(15, 35)  # sets the cubic-drag strength
            b = r(m * G / vt**3)
            v0 = r(rng.uniform(30, 70))
            ang = r(rng.uniform(25, 60), 2)
            truth = _range(v0, ang, G, drag=lambda s: b * s**3 / m)
            naive = v0**2 * np.sin(2 * np.radians(ang)) / G
            prompt = (f"In a hypothetical medium, the drag force on a moving object has magnitude "
                      f"b*v^3, directed opposite to the velocity, with b = {b} kg s/m^2. A ball of "
                      f"mass {m} kg is launched from flat ground at {v0} m/s at {ang} degrees above "
                      f"the horizontal. Take g = 9.81 m/s^2. How far away, in metres, does it land?")
            out.append(make_problem(FAMILY, level, i, prompt, truth, "m",
                                    {"m": m, "b": b, "v0": v0, "angle": ang}, naive))
    return out
