# Imports
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss
import copy
from numpy.linalg import norm, svd, solve
from scipy.optimize import fsolve
from legendre import legendre, legendre_derivative, gauss_legendre
from lanczos import lanczos_svd, kronecker_svd
from rungekutta import rk_scheme, RungeKuttaMethod
from mesh import StructuredMesh2D
from dofhandler import DoFHandler2D
from helpers import print_matrix, show, average, lagrange_basis, lagrange_basis_derivative

# Hyperparameters
dim = 2
courant_number = 0.2
flux_alpha = 1.0
start_time = 0
final_time = 1
output_tick = 0.1
mesh_type = "cartesian"
periodic = True
factor_skew = 0.5
use_gl_quad = False
use_gl_quad_mass = False
precontioner_type = "svd"
flux = "upwind"

# Hyperparameters grid
Nx = 1
Ny = 1
Lx = 1.0
Ly = 1.0
dx = Lx / Nx
dy = Ly / Ny

# Hyperparameters polynomials and quadrature
fe_degree = 2
Nq = fe_degree + 1
basis_type = "lagrange"

# Hyperparameters time integration
explicit = True
rk = 1
h = 0.1

# exact solution at time = 0
def initial_solution(x1, x2):
  return np.sin(2.0 * np.pi * x1) * np.sin(2.0 * np.pi * x2)

# exact analytical solution at any time
def exact_solution(x1, x2, t):
  return initial_solution(x1 - transport_speed()[0] * t, x2 - transport_speed()[1] * t)
  
# exact solution at boundary
def boundary_solution(x1, x2, t):
    # if x1 != 1 and x1 != -1 and x2 != 1 and x2 != -1:
    #     raise ValueError("no boundary term")
    return exact_solution(x1, x2, t)
  
# constant tranport speed in spacetime
def transport_speed():
    return np.array([1.0, 1.0])
        
# local operator
class CellWiseOperator:
  def __init__(self, mesh):
    """ Build operator for reference cell."""
    self.mesh = mesh
       
    """Precompute function values and derivatives on reference cell (volume and faces)."""
    self.qpts, self.qwts = gauss_legendre(Nq)
    
    self.quad_pts2D = [(self.qpts[k], self.qpts[l]) for k in range(Nq) for l in range(Nq)]
    self.quad_wts2D = [self.qwts[k] * self.qwts[l] for k in range(Nq) for l in range(Nq)]
    self.phi = np.zeros((Nq**2, Nq**2))
    self.dphi_dx = np.zeros((Nq**2, Nq**2, dim))
    self.bd_phi = dict()

    for i in range(Nq):
        for j in range(Nq):
            idx = i*Nq + j
            for q, (xq,yq) in enumerate(self.quad_pts2D):
                self.phi[q, idx] = self.basis_function(i, xq) * self.basis_function(j, yq)
                self.dphi_dx[q, idx, 0] = self.basis_function_derivative(i, xq) * self.basis_function(j, yq)
                self.dphi_dx[q, idx, 1] = self.basis_function(i, xq) * self.basis_function_derivative(j, yq)
                
            for face in ["left", "right", "top", "bottom"]:
                self.bd_phi[face] = np.zeros((Nq, Nq**2))
                for q, qpt in enumerate(self.qpts):
                    xq, yq = self.get_boundary_point(qpt, face)
                    self.bd_phi[face][q, idx] = self.basis_function(i, xq) * self.basis_function(i, yq) 
                    
    """Precompute 1D function values and derivatives to exploit tensor product structure."""
    self.phi1D = np.zeros((Nq, Nq))
    self.dphi1D = np.zeros((Nq, Nq))

    for i in range(Nq):
      for j in range(Nq):
        self.phi1D[j, i] = self.basis_function(i, self.qpts[j])
        self.dphi1D[j, i] = self.basis_function_derivative(i, self.qpts[j])

    self.wts1D = np.zeros((Nq, Nq))
    for i in range(Nq):
        self.wts1D[i, i] = self.qwts[i]
        
  def basis_function(self, i, x, type=basis_type):
    if type == "legendre":
        return legendre(i, x)
    else:
        return lagrange_basis(self.qpts, i, x)

  def basis_function_derivative(self, i, x, type=basis_type):
    if type == "legendre":
        return legendre_derivative(i, x)
    else:
        return lagrange_basis_derivative(self.qpts, i, x)
        
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

  def map_to_original(self, i, j, z1, z2):
      """Return mapping from reference to original cell"""
      x0, x1, y0, y1 = self.mesh.cell_bounds(i, j)
      return x0 + ((z1 + 1) / 2) * (x1 - x0), y0 + ((z2 + 1) / 2) * (y1 - y0)
  
  def map_to_reference(self, i, j, z1, z2):
      """Return mapping from original to reference cell"""
      x0, x1, y0, y1 = self.mesh.cell_bounds(i, j)
      return ((z1 - x0) / (x1 - x0) - 0.5) * 2, ((z2 - y0) / (y1 - y0) - 0.5) * 2
  
  def map_to_neighbor(self, face, z1, z2):
      """Return reference coordinates of neighbor cell"""
      if face == "left":
          return 1, z2
      elif face == "right":
          return -1, z2
      elif face == "top":
          return z1, -1
      elif face == "bottom":
          return z1, 1
      else:
          raise ValueError("invalid face")
  
  def jac(self, i, j):
    """Return the Jacobian transformation matrix for cell (i, j)."""
    x0, x1, y0, y1 = self.mesh.cell_bounds(i, j)
    jac = np.zeros((dim, dim))
    jac[0, 0] = 2.0 / (x1 - x0)
    jac[1, 1] = 2.0 / (y1 - y0)
    return jac
  
  def det_jac(self, i, j):
    """Return the determinant of Jacobian transformation matrix for cell (i, j)."""
    x0, x1, y0, y1 = self.mesh.cell_bounds(i, j)
    jac = (x1 - x0) * (y1 - y0) / 4
    return jac

  def det_jac_face(self, i, j, face):
    """Return the determinant of Jacobian transformation matrix for cell face (i, j)."""
    x0, x1, y0, y1 = self.mesh.cell_bounds(i, j)
    if face == "left":
        jac = (y1 - y0) / 2.0
    elif face == "right":
        jac = (y1 - y0) / 2.0
    elif face == "top":
        jac = (x1 - x0) / 2.0
    elif face == "bottom":
        jac = (x1 - x0) / 2.0
    else:
        raise ValueError("Invalid face")
    return jac

  def local_mass(self, ii, jj, tp = False):
    if tp:
        M_loc = np.kron(self.phi1D.T @ self.wts1D, self.phi1D.T @ self.wts1D) @ self.jac(ii, jj) @ np.kron(self.phi1D, self.phi1D)
    else:
        det_jac = self.det_jac(ii, jj)
        M_loc = np.zeros((Nq**2, Nq**2))
        for j in range(Nq**2):
            for i in range(Nq**2):
                for q, w in enumerate(self.quad_wts2D):
                    M_loc[j,i] += w * self.phi[q,j] * self.phi[q,i] * det_jac
        
    return M_loc

  def local_volume(self, ii, jj, tp = False):
    if tp:
        B_loc = (np.kron(self.phi1D.T @ self.wts1D, self.dphi1D.T @ self.wts1D) + np.kron(self.dphi1D.T @ self.wts1D, self.phi1D.T @ self.wts1D)) @ self.jac(ii, jj) @ np.kron(self.phi1D, self.phi1D)
    else:           
        a = transport_speed()
        det_jac = self.det_jac(ii, jj)
        jac = self.jac(ii, jj)
        B_loc= np.zeros((Nq**2, Nq**2))
        for j in range(Nq**2):
            for i in range(Nq**2):
                for q, w in enumerate(self.quad_wts2D):
                    B_loc[j,i] += w * self.phi[q,i] * np.dot(a, jac @ self.dphi_dx[q,j,:]) * det_jac
    return B_loc

  def local_f(self, ii, jj, t):
      # computes the right-hand side of the initial system
      det_jac = self.det_jac(ii, jj)
      F_loc = np.zeros((Nq**2, 1))               
      for i in range(Nq**2):
        for q, w in enumerate(self.quad_wts2D):
            x1 = self.quad_pts2D[q][0]
            x2 = self.quad_pts2D[q][1]
            x1, x2 = self.map_to_original(ii, jj, x1, x2)
            F_loc[i] += w * self.phi[q,i] * exact_solution(x1, x2, t) * det_jac
            # if exact_solution(x1, x2, t) > 10e-08:
            #     print(exact_solution(x1, x2, t))
      return F_loc
  
  def evaluate_function(self, U, x, y, dofs):
    val = 0.0
    for li, gi in enumerate(dofs):
        i = li // Nq
        j = li % Nq
        val += U[gi] * self.basis_function(i, x) * self.basis_function(j, y)

    return val

# global operator
class AdvectionOperator:
  def __init__(self, mesh, dofhandler, cellop, t):
        self.mesh = mesh
        self.dofhandler = dofhandler
        self.cellop = cellop
        self.ndofs = dofhandler.ndofs
        self.time = 0
        self.M = np.zeros((self.ndofs, self.ndofs))
        self.M_inv = np.zeros((self.ndofs, self.ndofs))
        self.B = np.zeros((self.ndofs, self.ndofs))
        self.G = np.zeros((self.ndofs, self.ndofs))
        self.Gbound = np.zeros((self.ndofs, 1))

  def assemble_system(self):
      # Loop over cells
      for (ii, jj), dofs in self.dofhandler.iter_cells():
          # local mass and volume
          M_loc = self.cellop.local_mass(ii, jj)
          B_loc = self.cellop.local_volume(ii, jj)
          M_loc_inv = np.linalg.inv(M_loc)

          # global mass and volume
          for li, gi in enumerate(dofs):
              for lj, gj in enumerate(dofs):
                  self.M[gi, gj] += M_loc[li, lj]
                  self.B[gi, gj] += B_loc[li, lj]
                  self.M_inv[gi, gj] += M_loc_inv[li, lj]

      # Loop over faces
      for face in self.dofhandler.iter_faces():
          dofs_cell = face["dofs_cell"]
          dofs_neighbor = face["dofs_neighbor"]
          normal = face["normal"]
          (ii, jj) = face["cell"]
          face_jac = self.cellop.det_jac_face(ii, jj, face["face"])
          if flux != "upwind":
              raise ValueError(f"Flux {flux} is not implemented.")
          
          if dofs_neighbor is None: # boundary
              for lj, gj in enumerate(dofs_cell):
                for q, qpt in enumerate(self.cellop.qpts):
                    normal_speed = transport_speed() @ normal               
                    xq, yq = self.cellop.get_boundary_point(qpt, face["face"])
                    qwt = self.cellop.qwts[q]
                    
                    if normal_speed < 0: # inflow  
                        x1, x2 = self.cellop.map_to_original(ii, jj, xq, yq)
                        self.Gbound[gj] += normal_speed * qwt * face_jac * self.cellop.bd_phi[face["face"]][q, lj] * boundary_solution(x1, x2, self.time)
                    else: # outflow
                        for li, gi in enumerate(dofs_cell):
                            self.G[gj, gi] += normal_speed * qwt * face_jac * self.cellop.bd_phi[face["face"]][q, lj] * self.cellop.bd_phi[face["face"]][q, li]
                                            
          else: # interior face             
              for lj, gj in enumerate(dofs_cell):
                for q, qpt in enumerate(self.cellop.qpts):
                    normal_speed = transport_speed() @ normal               
                    xq, yq = self.cellop.get_boundary_point(qpt, face["face"])
                    qwt = self.cellop.qwts[q]

                    if normal_speed < 0: # inflow  
                        for li, gi in enumerate(dofs_neighbor):
                            x1, x2 = self.cellop.map_to_original(ii, jj, xq, yq)
                            self.G[gj, gi] += normal_speed * qwt * face_jac *  self.cellop.bd_phi[face["face"]][q, lj] * self.cellop.bd_phi[face["face"]][q, li]
                    else: # outflow
                        for li, gi in enumerate(dofs_cell):
                            self.G[gj, gi] += normal_speed * qwt * face_jac *  self.cellop.bd_phi[face["face"]][q, lj] * self.cellop.bd_phi[face["face"]][q, li]
                            
  def set_time(self, t):
      # everything is time-independent except for Gbound, because the boundary condition depends on time
      if self.time == t:
          return
      self.time = t
      self.Gbound = np.zeros((self.ndofs, 1))
      # Loop over faces
      for face in self.dofhandler.iter_faces():
          dofs_cell = face["dofs_cell"]
          dofs_neighbor = face["dofs_neighbor"]
          normal = face["normal"]
          (ii, jj) = face["cell"]
          face_jac = self.cellop.det_jac_face(ii, jj, face["face"])
          if flux != "upwind":
              raise ValueError(f"Flux {flux} is not implemented.")
          
          if dofs_neighbor is None: # boundary
              for lj, gj in enumerate(dofs_cell):
                for q, qpt in enumerate(self.cellop.qpts):
                    normal_speed = transport_speed() @ normal               
                    xq, yq = self.cellop.get_boundary_point(qpt, face["face"])
                    qwt = self.cellop.qwts[q]
                    
                    if normal_speed < 0: # inflow  
                        x1, x2 = self.cellop.map_to_original(ii, jj, xq, yq)
                        self.Gbound[gj] += normal_speed * qwt * face_jac * self.cellop.bd_phi[face["face"]][q, lj] * boundary_solution(x1, x2, self.time)
      
  def apply(self, src):
      return self.M_inv @ ((self.B - self.G) @ src - self.Gbound)
  
  def interpolate(self, t):
      F = np.zeros((self.ndofs, 1))
      for (i, j), dofs in self.dofhandler.iter_cells():
          # local F
          F_loc = self.cellop.local_f(i, j, t)
          print_matrix(f"F_loc({i},{j})", F_loc)
          
          # global mass and volume
          for li, gi in enumerate(dofs):
            F[gi] += F_loc[li]
      return self.M_inv @ F
  
  def evaluate_function(self, U, x, y):
    for (i, j), dofs in self.dofhandler.iter_cells():
        x0, x1, y0, y1 = self.mesh.cell_bounds(i, j)
        if x0 <= x <= x1 and y0 <= y <= y1:
            return self.cellop.evaluate_function(U, x, y, dofs)

    raise ValueError("point outside domain")
  
def run():
    t = start_time
    mesh = StructuredMesh2D(Nx, Ny)
    dof_handler = DoFHandler2D(mesh, Nq**2)
    cell_operator = CellWiseOperator(mesh)
    operator = AdvectionOperator(mesh, dof_handler, cell_operator, t)
    operator.assemble_system()
    print_matrix("M", operator.M)
    print_matrix("M_inv", operator.M_inv)
    print_matrix("B", operator.B)
    print_matrix("G", operator.G)
    print_matrix("Gbound", operator.Gbound)
    
    rk_A, rk_b, rk_c = rk_scheme(rk=rk, explicit=explicit)
    rk_stepper = RungeKuttaMethod(rk_A, rk_b, rk_c)
    U = operator.interpolate(t)
    print_matrix("U", U)
    
    #show(lambda x, y: operator.evaluate_function(U, x, y))
    #show(lambda x, y: initial_solution(x, y))
    #show(lambda x, y: operator.evaluate_function(U, x, y) - initial_solution(x, y))
    error = []
    error.append(average(lambda x, y: operator.evaluate_function(U, x, y) - initial_solution(x, y)))
    print(f"Midpoint distance: {np.abs(operator.evaluate_function(U, 0.5, 0.5) - initial_solution(0.5, 0.5))}.")
    print(f"Average distance: {error[0]}")
    show(lambda x, y: operator.evaluate_function(U, x, y))
    
    def F(t, A):
        operator.set_time(t)
        return operator.apply(A)
    
    iter = 0
    while t < final_time:
        U = rk_stepper.step("", F, A0=U , h=h, t=t)
        t += h
        iter += 1
        error.append(average(lambda x, y: operator.evaluate_function(U, x, y) - exact_solution(x, y, t)))
        if iter % 1 == 0:
            print(f"Average error in iteration {iter}: {error[iter]}.")
            
    show(lambda x, y: operator.evaluate_function(U, x, y))
  
show(lambda x, y: exact_solution(x, y, 1))  
run()
    
# 1) make the explicit method run
    # check interpolation
    # check all matrices
    # check time stepping                
# 2) make the implicit method run
# 3) use the preconditioner for the implicit method
# 4) if it works, translate it to c++