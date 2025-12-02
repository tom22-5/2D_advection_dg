import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def print_matrix(name, mat):
    print(f"\n{name}: shape={mat.shape}")
    with np.printoptions(precision=4, suppress=True):
        print(mat)
        
def show(evaluate):
    # Grid size
    Nx, Ny = 100, 100

    # Domain [0,1] x [0,1]
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y)

    # Evaluate DG solution on grid
    Uplot = np.zeros_like(X)
    for i in range(Nx):
        for j in range(Ny):
            Uplot[j, i] = evaluate(X[j, i], Y[j, i])

    # ---- 3D Plot ----
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, Uplot, linewidth=0, antialiased=True)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u(x,y)")
    ax.set_title("DG Solution (3D Surface Plot)")

    plt.show()
    plt.pause(5)

def average(evaluate):
    Nx, Ny = 100, 100

    # Domain [0,1] x [0,1]
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y)

    sum = 0
    for i in range(Nx):
        for j in range(Ny):
            sum += np.abs(evaluate(X[j, i], Y[j, i]))

    sum = sum / (Nx * Ny)
    return sum

def lagrange_basis(x_nodes, i, x):
    """Return L_i(x) for Lagrange basis index i at point x"""
    val = 1.0
    xi = x_nodes[i]
    for j, xj in enumerate(x_nodes):
        if j != i:
            val *= (x - xj) / (xi - xj)
    return val


def lagrange_basis_derivative(x_nodes, i, x):
    """Return dL_i(x)/dx for basis index i at point x"""
    xi = x_nodes[i]
    N = len(x_nodes)
    total = 0.0
    
    for m in range(N):
        if m != i:
            term = 1.0 / (xi - x_nodes[m])
            for j in range(N):
                if j != i and j != m:
                    term *= (x - x_nodes[j]) / (xi - x_nodes[j])
            total += term
            
    return total
