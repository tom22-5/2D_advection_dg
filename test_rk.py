import numpy as np
import matplotlib.pyplot as plt
from rungekutta import RungeKuttaMethod, rk_scheme

def F(t, y):
    return 2*y

A, b, c = rk_scheme(4)
rk_method = RungeKuttaMethod(A, b, c)

# ---------- Simulation ----------
t0, tf, h = 0.0, 1.0, 0.2
N = int((tf - t0) / h)
t_vals = np.linspace(t0, tf, N + 1)

y = np.zeros(N + 1)
y[0] = 1.0

for i in range(N):
    y[i+1] = rk_method.step(None, F, y[i], h, i * h)

# exact solution
y_exact = np.exp(2 * t_vals)

# ---------- Plot ----------
plt.plot(t_vals, y, label="RK4 Numerical")
plt.plot(t_vals, y_exact, '--', label="Exact Solution")
plt.title("y' = 2y — explicit RK test")
plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()