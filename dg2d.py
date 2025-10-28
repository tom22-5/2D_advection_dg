# Imports
import math
import numpy as np
from numpy.polynomial.legendre import leggauss
import copy
from numpy.linalg import norm, svd, solve
from scipy.optimize import fsolve
from legendre import legendre, legendre_derivative, gauss_legendre
from lanczos import lanczos_svd, kronecker_svd
from rungekutta import rk_scheme, RungeKuttaMethod
from mesh import StructuredMesh2D
from dofhandler import DoFHandler2D

# Hyperparameters
dim = 2
courant_number = 0.2
flux_alpha = 1.0
start_time = 0
final_time = 8
output_tick = 0.1
mesh_type = "cartesian"
periodic = True
factor_skew = 0.5
use_gl_quad = False
use_gl_quad_mass = False
precontioner_type = "svd"
flux = "upwind"

# Hyperparameters grid
Nx = 20
Ny = 20
Lx = 1.0
Ly = 1.0
dx = Lx / Nx
dy = Ly / Ny

# Hyperparameters polynomials and quadrature
fe_degree = 1
Nq = fe_degree + 1

# Hyperparameters time integration
explicit = True
rk = 1
h = 0.01

# exact solution at time = 0
def initial_solution(x1, x2):
  return np.sin(2.0 * np.pi * x1) * np.sin(2.0 * np.pi * x2)

# exact analytical solution at any time
def exact_solution(x1, x2, t):
  return initial_solution(x1 - transport_speed()[0] * t, x2 - transport_speed()[1] * t)
  
# exact solution at boundary
def boundary_solution(x1, x2, t):
    if x1 != 1 and x1 != -1 and x2 != 1 and x2 != -1:
        raise ValueError("no boundary term")
    return exact_solution(x1, x2, t)
  
# constant tranport speed in spacetime
def transport_speed():
    return np.array([1.0, 1.0])

# local operator
class CellWiseOperator:
  def __init__(self, mesh):
    """ Build operator for reference cell."""
    self.mesh = mesh
    # precompute quadrature points and weights (1D)
    self.qpts, self.qwts = gauss_legendre(Nq)

    # mass matrix with tensor product
    self.phi1D = np.zeros((Nq, Nq))
    self.dphi1D = np.zeros((Nq, Nq))

    for i in range(Nq):
      for j in range(Nq):
        self.phi1D[j, i] = legendre(i, self.qpts[j]) # ith function at jth point
        self.dphi1D[j, i] = legendre_derivative(i, self.qpts[j]) # ith function at jth point

    self.wts1D = np.zeros((Nq, Nq))
    for i in range(Nq):
       self.wts1D[i, i] = self.qwts[i]

  def jac(self, i, j):
    """Return Jacobian matrix for cell (i, j)."""
    x0, x1, y0, y1 = self.mesh.cell_bounds(i, j)
    jac = np.zeros((Nq**2, Nq**2))
    for i in range(Nq**2):
        jac[i, i] = 4.0 / ((x1 - x0) * (y1 - y0))
    return jac

  def jac_face(self, i, j, face):
    """Return Jacobian matrix for cell face (i, j)."""
    x0, x1, y0, y1 = self.mesh.cell_bounds(i, j)
    jac = np.zeros((Nq, 1))
    for i in range(Nq):
        if face == "left":
            jac[i, 0] = 2.0 / (y1 - y0)
        elif face == "right":
            jac[i, 0] = 2.0 / (y1 - y0)
        elif face == "up":
            jac[i, 0] = 2.0 / (x1 - x0)
        elif face == "down":
            jac[i, 0] = 2.0 / (x1 - x0)
        else:
            raise ValueError("Invalid face")
    return jac

  def local_mass(self, i, j):
    M_loc = np.kron(self.phi1D.T @ self.wts1D, self.phi1D.T @ self.wts1D) @ self.jac(i, j) @ np.kron(self.phi1D, self.phi1D)
    return M_loc

  def local_volume(self, i, j):
    B_loc = (np.kron(self.phi1D.T @ self.wts1D, self.dphi1D.T @ self.wts1D) + np.kron(self.dphi1D.T @ self.wts1D, self.phi1D.T @ self.wts1D)) @ self.jac(i, j) @ np.kron(self.phi1D, self.phi1D)
    return B_loc

  def local_f(self, i, j, t):
      # Computes the right-hand side of the initial system
      F_loc = np.zeros((Nq**2, 1))
      for i in range(Nq):
        for j in range(Nq):
            for qpt in self.qpts:
                F_loc[i * Nq + j] += np.sum(legendre(i, self.qpts) * legendre(j, qpt) * self.qwts * self.jac(i, j)[0, 0] * exact_solution(self.qpts, qpt, t))
                
      return F_loc

# global operator
class AdvectionOperator:
  def __init__(self, mesh, dofhandler, cellop, t):
        self.mesh = mesh
        self.dofhandler = dofhandler
        self.cellop = cellop
        self.ndofs = dofhandler.ndofs
        self.t = 0
        self.M = np.zeros((self.ndofs, self.ndofs))
        self.M_inv = np.zeros((self.ndofs, self.ndofs))
        self.B = np.zeros((self.ndofs, self.ndofs))
        self.G = np.zeros((self.ndofs, self.ndofs))
        self.Gbound = np.zeros((self.ndofs, 1))
        
  def get_boundary_point(self, qpt, face):
      if face == "left":
          return -1, qpt
      elif face == "right":
          return 1, qpt
      elif face == "top":
          return qpt, 1
      elif face == "bottom":
          return qpt, -1
      else:
          raise ValueError("Invalid face")

  def assemble_system(self):
      # Loop over cells
      for (i, j), dofs in self.dofhandler.iter_cells():
          # local mass and volume
          M_loc = self.cellop.local_mass(i, j)
          B_loc = self.cellop.local_volume(i, j)

          # global mass and volume
          for li, gi in enumerate(dofs):
              for lj, gj in enumerate(dofs):
                  self.M[gi, gj] += M_loc[li, lj]
                  self.B[gi, gj] += B_loc[li, lj]

      # Loop over faces
      for face in self.dofhandler.iter_faces():
          dofs_cell = face["dofs_cell"]
          dofs_neighbor = face["dofs_neighbor"]
          normal = face["normal"]
          (i, j) = face["cell"]
          if flux != "upwind":
              raise ValueError(f"Flux {flux} is not implemented.")
          
          if dofs_neighbor is None: # boundary
              for lj, gj in enumerate(dofs_cell):
                lj1 = lj % Nq 
                lj2 = lj - lj1 * Nq
            
                for qpt in self.cellop.qpts: 
                    pt = self.get_boundary_point(qpt, face["face"])
                    if transport_speed() @ normal > 0: # inflow  
                        self.Gbound[gj] += (transport_speed() @ normal) * self.cellop.qwts * self.cellop.jac_face(i, j, face["face"]) * legendre(lj1, pt[0]) * legendre(lj2, pt[1]) * boundary_solution(pt[0], pt[1], self.t)
                    else: # outflow
                        for li, gi in enumerate(dofs_cell):
                            li1 = li % Nq
                            li2 = li - li1 * Nq
                            self.G[gj, gi] += (transport_speed() @ normal) * self.cellop.qwts * self.cellop.jac_face(i, j, face["face"]) * legendre(lj1, pt[0]) * legendre(lj2, pt[1]) * legendre(li1, pt[0]) * legendre(li2, pt[1]) 
                
          else: # interior face             
              for lj, gj in enumerate(dofs_cell):
                lj1 = lj % Nq 
                lj2 = lj - lj1 * Nq
                  
                for qpt in self.cellop.qpts: 
                    pt = self.get_boundary_point(qpt, face["face"])

                    if transport_speed() @ normal > 0: # inflow  
                        for li, gi in enumerate(dofs_neighbor):
                            li1 = li % Nq
                            li2 = li - li1 * Nq
                            self.G[gj, gi] += (transport_speed() @ normal) * self.cellop.qwts * self.cellop.jac_face(i, j, face["face"]) * legendre(lj1, pt[0]) * legendre(lj2, pt[1]) * legendre(li1, pt[0]) * legendre(li2, pt[1])
                    else: # outflow
                        for li, gi in enumerate(dofs_cell):
                            li1 = li % Nq
                            li2 = li - li1 * Nq
                            self.G[gj, gi] += (transport_speed() @ normal) * self.cellop.qwts * self.cellop.jac_face(i, j, face["face"]) * legendre(lj1, pt[0]) * legendre(lj2, pt[1]) * legendre(li1, pt[0]) * legendre(li2, pt[1])
                            
      self.M_inv = np.linalg.inv(self.M)
  
  def set_time(self, t):
      # everything is time-independent except for Gbound, because the boundary condition depends on time
      self.time = t
      self.Gbound = np.zeros((self.ndofs, 1))
      # Loop over faces
      for face in self.dofhandler.iter_faces():
          dofs_cell = face["dofs_cell"]
          dofs_neighbor = face["dofs_neighbor"]
          normal = face["normal"]
          (i, j) = face["cell"]
          if flux != "upwind":
              raise ValueError(f"Flux {flux} is not implemented.")
          
          if dofs_neighbor is None: # boundary
              for lj, gj in enumerate(dofs_cell):
                lj1 = lj % Nq 
                lj2 = lj - lj1 * Nq
            
                for qpt in self.cellop.qpts: 
                    pt = self.get_boundary_point(qpt, face["face"])
                    if transport_speed() @ normal > 0: # inflow  
                        # todo: shapes don't match
                        self.Gbound[gj] += (transport_speed() @ normal) * self.cellop.qwts * self.cellop.jac_face(i, j, face["face"]) * legendre(lj1, pt[0]) * legendre(lj2, pt[1]) * boundary_solution(pt[0], pt[1], self.t)
      
  def apply(self, src):
      return self.M_inv @ ((self.B.T - self.G) @ src + self.Gbound)
  
  def interpolate(self, t):
      F = np.zeros((self.ndofs, 1))
      for (i, j), dofs in self.dofhandler.iter_cells():
          # local F
          F_loc = self.cellop.local_f(i, j, t)

          # global mass and volume
          for li, gi in enumerate(dofs):
            F[gi] = F_loc[li]
                  
      return self.M_inv @ F
  
def run():
    t = start_time
    mesh = StructuredMesh2D(Nx, Ny)
    dof_handler = DoFHandler2D(mesh, Nq**2)
    cell_operator = CellWiseOperator(mesh)
    operator = AdvectionOperator(mesh, dof_handler, cell_operator, t)
    
    rk_A, rk_b, rk_c = rk_scheme(rk=rk, explicit=explicit)
    rk_stepper = RungeKuttaMethod(rk_A, rk_b, rk_c)
    U = operator.interpolate(t)
    U_perfect = operator.interpolate(t)
    error = [np.linalg.norm(U - U_perfect)]
    
    while t < final_time:
        U = rk_stepper.step(operator, A0=U , h=h, t=t)
        t += h
        U_perfect = operator.interpolate(t)
        error.append(np.norm(U - U_perfect))
        
    print(U)
    print(U_perfect)
    print(error)
    
run()
    
# 1) make the explicit method run
# 2) make the implicit method run
# 3) use the preconditioner for the implicit method
# 4) if it works, translate it to c++