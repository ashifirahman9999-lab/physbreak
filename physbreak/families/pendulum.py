"""Pendulum family. Ground truth always comes from numerical integration."""
import numpy as np
from scipy.integrate import solve_ivp

from .common import make_problem, r

FAMILY = "pendulum"
G = 9.81


def _period(accel, theta0):
    """Period of a symmetric oscillator theta'' = accel(theta), released from rest at theta0.

    Integrates until theta first crosses zero (a quarter period) and multiplies by 4.
    """
    def rhs(t, y):
        return [y[1], accel(y[0])]

    def cross(t, y):
        return y[0]
    cross.terminal = True
    cross.direction = -1

    sol = solve_ivp(rhs, [0, 1e4], [theta0, 0.0], events=cross,
                    rtol=1e-10, atol=1e-12, max_step=0.05)
    return 4.0 * sol.t_events[0][0]


def _double_pendulum_theta2(th1, th2, t_end, g=G, l=1.0):
    """Lower-rod angle (deg, wrapped to (-180, 180]) of an equal-mass, equal-length double pendulum."""
    def rhs(t, y):
        t1, t2, w1, w2 = y
        d = t1 - t2
        den = 3.0 - np.cos(2 * d)  # (2m1 + m2 - m2 cos 2d) with m1 = m2 = 1
        a1 = (-3 * g * np.sin(t1) - g * np.sin(t1 - 2 * t2)
              - 2 * np.sin(d) * (w2**2 * l + w1**2 * l * np.cos(d))) / (l * den)
        a2 = (2 * np.sin(d) * (2 * w1**2 * l + 2 * g * np.cos(t1)
              + w2**2 * l * np.cos(d))) / (l * den)
        return [w1, w2, a1, a2]

    sol = solve_ivp(rhs, [0, t_end], [th1, th2, 0.0, 0.0],
                    rtol=1e-10, atol=1e-12, method="DOP853")
    ang = np.degrees(sol.y[1, -1])
    return (ang + 180.0) % 360.0 - 180.0


def _circ_diff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def generate(level, n, rng):
    out = []
    for i in range(n):
        if level == 0:
            L = r(rng.uniform(0.5, 3.0))
            a0 = r(rng.uniform(2, 8), 2)
            truth = _period(lambda th: -(G / L) * np.sin(th), np.radians(a0))
            naive = 2 * np.pi * np.sqrt(L / G)
            prompt = (f"A simple pendulum of length {L} m is released from rest at an angle of "
                      f"{a0} degrees from the vertical. Take g = 9.81 m/s^2. "
                      f"What is its period of oscillation in seconds?")
            out.append(make_problem(FAMILY, level, i, prompt, truth, "s",
                                    {"L": L, "amp_deg": a0}, naive))
        elif level == 1:
            L = r(rng.uniform(1.0, 5.0))
            gp = r(rng.uniform(2.0, 15.0))
            a0 = r(rng.uniform(2, 8), 2)
            truth = _period(lambda th: -(gp / L) * np.sin(th), np.radians(a0))
            naive = 2 * np.pi * np.sqrt(L / gp)
            prompt = (f"A small bead is fixed to the end of a light rigid rod of length {L} spans. "
                      f"The other end of the rod is on a frictionless pivot, and the bead sits in a "
                      f"uniform downward pull of strength {gp} spans per tick squared. The bead is "
                      f"nudged aside so the rod makes {a0} degrees with the direction of the pull, "
                      f"then let go. How many ticks pass before the bead first returns to its "
                      f"release point?")
            out.append(make_problem(FAMILY, level, i, prompt, truth, "ticks",
                                    {"L": L, "g": gp, "amp_deg": a0}, naive))
        elif level == 2:
            L = r(rng.uniform(0.5, 3.0))
            a0 = r(rng.uniform(120, 170), 3)
            truth = _period(lambda th: -(G / L) * np.sin(th), np.radians(a0))
            naive = 2 * np.pi * np.sqrt(L / G)
            prompt = (f"A pendulum consisting of a small heavy bob on a light rigid rod of length "
                      f"{L} m is released from rest at an angle of {a0} degrees from the "
                      f"downward vertical. Take g = 9.81 m/s^2. What is its period of oscillation "
                      f"in seconds?")
            out.append(make_problem(FAMILY, level, i, prompt, truth, "s",
                                    {"L": L, "amp_deg": a0}, naive))
        elif level == 3:
            k = r(rng.uniform(2.0, 20.0))
            A = r(rng.uniform(0.3, 1.2))
            truth = _period(lambda th: -k * th**3, A)
            naive = 2 * np.pi / np.sqrt(k)  # treating it as a linear oscillator
            prompt = (f"In a hypothetical universe, a pendulum obeys the equation of motion "
                      f"d^2(theta)/dt^2 = -k * theta^3 exactly, where theta is in radians and "
                      f"k = {k} rad^-2 s^-2. It is released from rest at theta = {A} rad. "
                      f"What is its period of oscillation in seconds?")
            out.append(make_problem(FAMILY, level, i, prompt, truth, "s",
                                    {"k": k, "amp_rad": A}, naive))
        elif level == 4:
            # Half the cases are low-energy (predictable), half high-energy (chaotic),
            # so a model cannot score by always answering UNPREDICTABLE.
            chaotic_case = (i % 2 == 0)
            lo, hi = (100, 170) if chaotic_case else (3, 10)
            t1 = r(rng.uniform(lo, hi), 3)
            t2 = r(rng.uniform(lo, hi), 3)
            T = 30.0
            nominal = _double_pendulum_theta2(np.radians(t1), np.radians(t2), T)
            spread = 0.0
            for d1 in (-0.1, 0.1):
                for d2 in (-0.1, 0.1):
                    v = _double_pendulum_theta2(np.radians(t1 + d1), np.radians(t2 + d2), T)
                    spread = max(spread, _circ_diff(v, nominal))
            truth = "UNPREDICTABLE" if spread > 10.0 else nominal
            prompt = (f"A double pendulum consists of two identical point masses and two identical "
                      f"light rigid rods of length 1 m. Take g = 9.81 m/s^2. It is released from rest "
                      f"with the upper rod at {t1} degrees and the lower rod at {t2} degrees from the "
                      f"downward vertical (both measured the same way). The initial angles are only "
                      f"known to within +/-0.1 degrees. What is the angle of the lower rod from the "
                      f"downward vertical, in degrees in the range (-180, 180], after exactly "
                      f"{T:.0f} s? If the answer cannot be determined to within +/-10 degrees given "
                      f"the measurement uncertainty, answer \"UNPREDICTABLE\".")
            out.append(make_problem(FAMILY, level, i, prompt, truth, "deg",
                                    {"theta1_deg": t1, "theta2_deg": t2, "t": T,
                                     "spread_deg": spread},
                                    None, tol=10.0, tol_type="abs"))
    return out
