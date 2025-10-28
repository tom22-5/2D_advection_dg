import math
import numpy as np
from numpy.polynomial.legendre import leggauss
import copy
from numpy.linalg import norm, svd, solve
from scipy.optimize import fsolve


# basis and quadrature
def legendre(n, x):
    """Evaluate nth Legendre polynomial at x (recurrence)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x
    else:
        Pn_1 = x
        Pn_2 = np.ones_like(x)
        for k in range(2, n+1):
            Pn = ((2*k-1)*x*Pn_1 - (k-1)*Pn_2)/k
            Pn_2, Pn_1 = Pn_1, Pn
        return Pn

def legendre_derivative(n, x):
    """Derivative of nth Legendre polynomial at x."""
    if n == 0:
        return np.zeros_like(x)
    return n/(1 - x**2) * (legendre(n-1, x) - x*legendre(n, x))

# Quadrature
def gauss_legendre(n):
    """Gauss-Legendre quadrature points and weights."""
    return leggauss(n)