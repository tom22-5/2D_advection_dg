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
from scipy.sparse.linalg import gmres, LinearOperator
from lanczos import form_2d_preconditioner, apply_2d_preconditioner, lanczos_svd, kronecker_svd
import time

# Hyperparameters
dim = 2
courant_number = 0.2
flux_alpha = 1.0
start_time = 0
final_time = 0.1
output_tick = 0.1
mesh_type = "cartesian"
precontioner_type = "svd"
flux = "upwind"

# Hyperparameters grid
Nx = 10
Ny = 10
Lx = 1.0
Ly = 1.0
dx = Lx / Nx
dy = Ly / Ny

# Hyperparameters polynomials and quadrature
fe_degree = 3
Nq = fe_degree + 1
basis_type = "lagrange"

# Hyperparameters time integration
explicit = False
preconditioner = "pazner"
rk = 1
h = 0.001

# exact solution at time = 0
def initial_solution(x1, x2):
    # return 1
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
    self.bd_phi = {face: np.zeros((Nq, Nq**2)) for face in ["left", "right", "top", "bottom"]}

    for i in range(Nq):
        for j in range(Nq):
            idx = i*Nq + j
            for q, (xq,yq) in enumerate(self.quad_pts2D):
                self.phi[q, idx] = self.basis_function(i, xq) * self.basis_function(j, yq)
                self.dphi_dx[q, idx, 0] = self.basis_function_derivative(i, xq) * self.basis_function(j, yq)
                self.dphi_dx[q, idx, 1] = self.basis_function(i, xq) * self.basis_function_derivative(j, yq)
                
            for face in ["left", "right", "top", "bottom"]:
                for q, qpt in enumerate(self.qpts):
                    xq, yq = self.get_boundary_point(qpt, face)
                    self.bd_phi[face][q, idx] = self.basis_function(i, xq) * self.basis_function(j, yq) 
                    
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
  
  def opposite(self, face):
      """Return corresponding face for neighbor cell (opposite)"""
      if face == "left":
          return "right"
      elif face == "right":
          return "left"
      elif face == "top":
          return "bottom"
      elif face == "bottom":
          return "top"
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
                            face_neighbor = self.cellop.opposite(face["face"])
                            self.G[gj, gi] += normal_speed * qwt * face_jac *  self.cellop.bd_phi[face["face"]][q, lj] * self.cellop.bd_phi[face_neighbor][q, li]
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
  
  def apply_volume(self, src):
      return self.B @ src
  
  def apply_boundary(self, src):
      return self.G @ src
  
  def add_boundary(self):
      return self.Gbound
  
  def interpolate(self, t):
      F = np.zeros((self.ndofs, 1))
      for (i, j), dofs in self.dofhandler.iter_cells():
          # local F
          F_loc = self.cellop.local_f(i, j, t)
          #   print_matrix(f"F_loc({i},{j})", F_loc)
          
          # global mass and volume
          for li, gi in enumerate(dofs):
            F[gi] += F_loc[li]
      return self.M_inv @ F
  
  def evaluate_function(self, U, x, y):
    
    for (i, j), dofs in self.dofhandler.iter_cells():
        x0, x1, y0, y1 = self.mesh.cell_bounds(i, j)
        if x0 <= x <= x1 and y0 <= y <= y1:
            z1, z2 = self.cellop.map_to_reference(i, j, x, y)
            return self.cellop.evaluate_function(U, z1, z2, dofs)

    raise ValueError("point outside domain")
  
  def build_operator(self, factor):
    n = self.ndofs

    def matvec(v_flat):
        v = v_flat.reshape(n, 1)
        w = v - factor * (
            self.M_inv @ ((self.B - self.G) @ v)
        )
        return w.reshape(n)
    
    def rmatvec(v_flat):
        v = v_flat.reshape(n, 1)
        w = v - factor * (
            (self.B.T - self.G.T) @ self.M_inv.T @ v
        )
        return w.reshape(n)

    linear_operator = LinearOperator(
        shape=(n, n),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=float
    )
    return linear_operator
  
  def build_preconditioner(self, factor):
    precon_dicts = {}

    for (ii, jj), dofs in self.dofhandler.iter_cells():
        e = (ii, jj)
        dofs = np.array(dofs)
        Mloc = self.M_inv[np.ix_(dofs, dofs)]
        Bloc = self.B    [np.ix_(dofs, dofs)]
        Gloc = self.G    [np.ix_(dofs, dofs)]
        
        def loc_matvec(v_loc):
            return v_loc - factor * (
                Mloc @ ((Bloc - Gloc) @ v_loc)
            )
                    
        local_operator = LinearOperator(
            shape=(Nq**2, Nq**2),
            matvec=loc_matvec,
            dtype=float
        )

        A_list, B_list, _ = kronecker_svd(local_operator, Nq, Nq, Nq, Nq, r=2)
        # A_approx = sum(np.kron(A_list[j], B_list[j]) for j in range(2))
        # err = np.linalg.norm(A_approx - np.eye(Nq**2) - factor * self.M_inv[np.ix_(dofs, dofs)] @ (self.B[np.ix_(dofs, dofs)] - self.G[np.ix_(dofs, dofs)]))
        # print(f"Error for element {e}: {err}.")
        precon_dicts[e] = form_2d_preconditioner(local_operator, Nq, Nq, Nq, Nq, 2)
        
    def matvec(v):
        v = v.reshape(self.ndofs)
        w = np.zeros(self.ndofs)
        
        for (ii, jj), dofs in self.dofhandler.iter_cells():
            e = (ii, jj)
            v_local = v[dofs]
            w[dofs] = apply_2d_preconditioner(v_local, precon_dicts[e])
        
        return w

    return LinearOperator(shape=(self.ndofs, self.ndofs),matvec=matvec, dtype=float)

def run():
    t = start_time
    mesh = StructuredMesh2D(Nx, Ny)
    dof_handler = DoFHandler2D(mesh, Nq**2)
    cell_operator = CellWiseOperator(mesh)
    operator = AdvectionOperator(mesh, dof_handler, cell_operator, t)
    operator.assemble_system()
    # print_matrix("M", operator.M)
    # print_matrix("M_inv", operator.M_inv)
    # print_matrix("B", operator.B)
    # print_matrix("G", operator.G)
    # print_matrix("Gbound", operator.Gbound)
    
    rk_A, rk_b, rk_c = rk_scheme(rk=rk, explicit=explicit)
    rk_stepper = RungeKuttaMethod(rk_A, rk_b, rk_c)
    U = operator.interpolate(t)
    # print_matrix("U", U)
    
    #show(lambda x, y: operator.evaluate_function(U, x, y))
    #show(lambda x, y: initial_solution(x, y))
    #show(lambda x, y: operator.evaluate_function(U, x, y) - initial_solution(x, y))
    error = []
    error.append(average(lambda x, y: operator.evaluate_function(U, x, y) - initial_solution(x, y)))
    print(f"Midpoint distance: {np.abs(operator.evaluate_function(U, 0.5, 0.5) - initial_solution(0.5, 0.5))}.")
    print(f"Average distance: {error[0]}")
    # show(lambda x, y: operator.evaluate_function(U, x, y))
    
    def F(t, A):
        operator.set_time(t)
        return operator.apply(A)
    
    # test 4: integration
    # # Forward Euler reference
    # RHS = F(0.0, U)
    # U_FE = U + h * RHS

    # # RK4 step
    # rk_A, rk_b, rk_c = rk_scheme(rk=4, explicit=True)
    # rk_stepper = RungeKuttaMethod(rk_A, rk_b, rk_c)
    # U_RK4 = rk_stepper.step("", F, A0=U, h=h, t=0.0) 
    # diff_FE_RK4 = U_FE - U_RK4
    # print("||U_FE - U_RK4|| =", norm(diff_FE_RK4))
    # return
    
    iter = 0
    while t < final_time:
        U = rk_stepper.step(operator, F, A0=U , h=h, t=t, precondition=False)
        t += h
        iter += 1
        error.append(average(lambda x, y: operator.evaluate_function(U, x, y) - exact_solution(x, y, t)))
        # show(lambda x, y: operator.evaluate_function(U, x, y)) 
        # show(lambda x, y: exact_solution(x, y, t)) 
        if iter % 1 == 0:
            print(f"Average error in iteration {iter}: {error[iter]}.")
            
    # show(lambda x, y: operator.evaluate_function(U, x, y))
  
### tests
# test 1: constant solution
def test_constant_solution():
    t = 0.0
    mesh = StructuredMesh2D(Nx, Ny)
    dof_handler = DoFHandler2D(mesh, Nq**2)
    cell_operator = CellWiseOperator(mesh)
    op = AdvectionOperator(mesh, dof_handler, cell_operator, t)
    op.assemble_system()
    test_components(op)
    inspect_G_on_constant(op)

    # constant state
    U = np.ones((op.ndofs, 1))

    op.set_time(t)
    RHS = op.apply(U)   # should be ~0 for all entries

    print("||RHS(constant)|| =", norm(RHS))

def test_components(op):
    # constant state
    U = np.ones((op.ndofs, 1))
    
    vol = op.apply_volume(U)
    surf = op.apply_boundary(U)
    bd = op.add_boundary()
    
    print("||B e|| =", norm(vol))
    print("||G e|| =", norm(surf))
    print("||Gbound|| =", norm(bd))

def inspect_G_on_constant(op):
    e = np.ones((op.ndofs, 1))
    ge = op.G @ e

    print("Global ||G e|| =", norm(ge))

    # print by cell
    for (ii, jj), dofs in op.dofhandler.iter_cells():
        cell_vals = ge[dofs].ravel()
        print(f"cell ({ii},{jj})  G e =", cell_vals)
        
    print(f"Global Gbound: {op.Gbound}")

# test 2: time derivative for 2D sine
def test_time_derivative():
    t = 0.0
    mesh = StructuredMesh2D(Nx, Ny)
    dof_handler = DoFHandler2D(mesh, Nq**2)
    cell_operator = CellWiseOperator(mesh)
    op = AdvectionOperator(mesh, dof_handler, cell_operator, t)
    op.assemble_system()
    Ut_exact_proj = project_exact_ut(op)
    U = op.interpolate(t)

    op.set_time(t)
    RHS = op.apply(U)

    diff = RHS - Ut_exact_proj
    print("||RHS||           =", norm(RHS))
    print("||Ut_exact_proj|| =", norm(Ut_exact_proj))
    print("||diff||          =", norm(diff))
    print("Relative error    =", norm(diff)/norm(Ut_exact_proj))
    
def exact_ut(x, y):
    # u0 = sin(2πx) sin(2πy)
    # ∂x u0 = 2π cos(2πx) sin(2πy)
    # ∂y u0 = 2π sin(2πx) cos(2πy)
    return -(1.0 * (2*np.pi * np.cos(2*np.pi*x) * np.sin(2*np.pi*y))
             + 1.0 * (2*np.pi * np.sin(2*np.pi*x) * np.cos(2*np.pi*y)))

def project_exact_ut(operator):
    F = np.zeros((operator.ndofs, 1))
    for (ii, jj), dofs in operator.dofhandler.iter_cells():
        det_jac = operator.cellop.det_jac(ii, jj)
        F_loc = np.zeros((Nq**2, 1))
        for i in range(Nq**2):
            for q, w in enumerate(operator.cellop.quad_wts2D):
                z1, z2 = operator.cellop.quad_pts2D[q]
                x, y = operator.cellop.map_to_original(ii, jj, z1, z2)
                F_loc[i] += w * operator.cellop.phi[q, i] * exact_ut(x, y) * det_jac
        for li, gi in enumerate(dofs):
            F[gi] += F_loc[li]
    return operator.M_inv @ F

# test 3: time integration
def test_time_integrator():
    def F_test(t, u):
        return np.array([[1.0]])  # same shape as u

    rk_A, rk_b, rk_c = rk_scheme(rk=4, explicit=True)
    rk_stepper = RungeKuttaMethod(rk_A, rk_b, rk_c)

    u = np.array([[0.0]])
    t = 0.0
    h = 0.001

    for n in range(10):
        u = rk_stepper.step("", F_test, A0=u, h=h, t=t)
        t += h
        print(n+1, " t =", t, " u =", u[0,0], " exact =", t)
    
# test 4: test kronecker svd
def test_preconditioner():
    # test_lanczos()
    # test_kronecker()
    test_application()

def test_lanczos():
    # Generate a matrix
    A = np.array([
        [10, 0, 0, 0, 0],
        [0, 9, 0, 0, 0],
        [0, 0, 8, 0, 0],
        [0, 0, 0, 7, 0],
        [0, 0, 0, 0, 6],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    

    # Check its rank to be sure
    rank = np.linalg.matrix_rank(A)
    print("Rank of A:", rank)
    r = 5
    J = 11

    # Lanczos SVD (your method)
    linear_operator = LinearOperator(
        shape=A.shape,
        matvec=lambda x: A @ x,
        rmatvec=lambda x: A.T @ x,
        dtype=float
    )  
    U, sigma, V = lanczos_svd(linear_operator, J=J, r=r, tol=1e-10)
    # print(U.shape)
    # print(sigma.shape)
    # print(V.shape)

    # Reconstruct A from Lanczos SVD
    A_lanczos = U @ np.diag(sigma) @ V
    error_lanczos = np.linalg.norm(A - A_lanczos)

    print(f"Lanczos{J} reconstruction error: ", error_lanczos)

    print(U)
    print(sigma)
    print(V)
    # Full SVD for comparison
    U_opt, sigma_opt, V_opt = svd(A, full_matrices=False)
    A_opt = U_opt[:, :r] @ np.diag(sigma_opt[:r]) @ V_opt[:r, :]
    error_opt = np.linalg.norm(A - A_opt)
    print(U_opt)
    print(sigma_opt)
    print(V_opt)
    print("Optimal SVD reconstruction error: ", error_opt)

def test_kronecker():
    # Build a matrix A as a sum of Kronecker products
    m1, m2 = 2, 3
    n1, n2 = 2, 4
    r_true = 4

    A_true = sum(np.kron(np.random.randn(m1, n1), np.random.randn(m2, n2)) for _ in range(r_true))

    A0 = np.array([[1, 2],
                [3, 4]])
    B0 = np.array([[0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11]])

    A1 = np.array([[5, 6],
                [7, 8]])
    B1 = np.array([[1, 0, -1, 0],
                [0, 1, 0, -1],
                [1, 1, 1, 1]])

    # Kronecker sum
    A_true = np.kron(A0, B0) + np.kron(A1, B1)
    print(A_true.shape)
    print(A_true)

    A = LinearOperator(
        shape=A_true.shape,
        matvec=lambda x: A_true @ x,
        rmatvec=lambda x: A_true.T @ x,
        dtype=float
    )  
    
    # Step 1: Build Tilde Operator
    def matvec(v):        
        unit_vector = np.zeros(n1 * n2)
        solution_vector = np.zeros(m1 * n1)
        for j1 in range(n1):
            for j2 in range(n2):
                unit_vector[j1 * n2 + j2] = 1
                column_vector = A.matvec(unit_vector)
                unit_vector[j1 * n2 + j2] = 0
                
                Z = column_vector.reshape(m2, m1, order="F")
                for i1 in range(m1):
                    out_index = i1 + j1 * m1
                    for i2 in range(m2):
                        in_index = i2 + j2 * m2
                        solution_vector[out_index] += Z[i2, i1] * v[in_index]
        return solution_vector
    
    def rmatvec(v):        
        unit_vector = np.zeros(n1 * n2)
        solution_vector = np.zeros(m2 * n2)
        for j1 in range(n2):
            for j2 in range(n1):
                unit_vector[j2 * n2 + j1] = 1
                column_vector = A.matvec(unit_vector)
                unit_vector[j2 * n2 + j1] = 0
                
                Z = column_vector.reshape(m2, m1, order="F")
                for i2 in range(m2):
                    out_index = i2 + j1 * m2
                    for i1 in range(m1):
                        in_index = i1 + j2 * m1
                        solution_vector[out_index] += Z[i2, i1] * v[in_index]
        
        return solution_vector

    A_tilde = LinearOperator(
        shape=(m1 * n1, m2 * n2),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=float
    )
    
    # Build Tilde Matrix
    A_tilde_matrix = np.zeros((m1 * n1, m2 * n2))
    for i1 in range(m1):
        for i2 in range(n1):
            for j1 in range(n2):
                for j2 in range(m2):
                    A_tilde_matrix[i2 * m1 + i1, j1 * m2 + j2] = A_true[i1 * m2 + j2, i2 * n2 + j1]
    
    print_matrix("A", A_true)
    print_matrix("Atilde", A_tilde_matrix)
    
    vec = np.random.randn(m2 * n2)
    print(f"Atilde matvec error: {norm(A_tilde_matrix @ vec - A_tilde.matvec(vec))}")
    vec = np.random.randn(m1 * n1)
    print(f"Atilde rmatvec error: {norm(A_tilde_matrix.T @ vec - A_tilde.rmatvec(vec))}")
    pass

    # Compute KSVD
    A_list_opt, B_list_opt, s_opt = kronecker_svd(A, m1, m2, n1, n2, r=r_true, method="lanczos")

    r_true = min(len(A_list_opt), r_true)
    # Reconstruct approximation
    A_approx_opt = sum(np.kron(A_list_opt[j], B_list_opt[j]) for j in range(r_true))
    #A_approx = sum(np.kron(A_list[j], B_list[j]) for j in range(r_true))

    # Compare
    error = np.linalg.norm(A_true - A_approx_opt)
    # error_lanczos = np.linalg.norm(A_true - A_approx)
    print("Reconstruction error SVD:", error)
    print("Singular values from reshaped SVD:", s_opt)
    # print("Reconstruction error Lanczos SVD:", error_lanczos)
    # print("Singular values from Lanczos SVD:", s)
    
def test_application():
    # Force ALL sizes = 2
    m1, m2 = 2, 2
    n1, n2 = 2, 2
    r_true = 2

    # Build a known A_true = kron(A0, B0) + kron(A1, B1)
    A0 = np.array([[1, 2],
                   [3, 4]])

    B0 = np.array([[ 0,  1],
                   [ 2,  3]])

    A1 = np.array([[5, 6],
                   [7, 8]])

    B1 = np.array([[ 1,  0],
                   [-1,  1]])

    A_true = np.kron(A0, B0) + np.kron(A1, B1)

    print_matrix("A_true", A_true)

    # Linear operator for A
    A = LinearOperator(
        shape=A_true.shape,
        matvec=lambda x: A_true @ x,
        rmatvec=lambda x: A_true.T @ x,
        dtype=float
    )  
    
    # Step 1: Build Tilde Operator
    def matvec(v):        
        unit_vector = np.zeros(n1 * n2)
        solution_vector = np.zeros(m1 * n1)
        for j1 in range(n1):
            for j2 in range(n2):
                unit_vector[j1 * n2 + j2] = 1
                column_vector = A.matvec(unit_vector)
                unit_vector[j1 * n2 + j2] = 0
                
                Z = column_vector.reshape(m2, m1, order="F")
                for i1 in range(m1):
                    out_index = i1 + j1 * m1
                    for i2 in range(m2):
                        in_index = i2 + j2 * m2
                        solution_vector[out_index] += Z[i2, i1] * v[in_index]
        return solution_vector
    
    def rmatvec(v):        
        unit_vector = np.zeros(n1 * n2)
        solution_vector = np.zeros(m2 * n2)
        for j1 in range(n2):
            for j2 in range(n1):
                unit_vector[j2 * n2 + j1] = 1
                column_vector = A.matvec(unit_vector)
                unit_vector[j2 * n2 + j1] = 0
                
                Z = column_vector.reshape(m2, m1, order="F")
                for i2 in range(m2):
                    out_index = i2 + j1 * m2
                    for i1 in range(m1):
                        in_index = i1 + j2 * m1
                        solution_vector[out_index] += Z[i2, i1] * v[in_index]
        
        return solution_vector

    A_tilde = LinearOperator(
        shape=(m1 * n1, m2 * n2),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=float
    )
    
    # Build Tilde Matrix
    A_tilde_matrix = np.zeros((m1 * n1, m2 * n2))
    for i1 in range(m1):
        for i2 in range(n1):
            for j1 in range(n2):
                for j2 in range(m2):
                    A_tilde_matrix[i2 * m1 + i1, j1 * m2 + j2] = A_true[i1 * m2 + j2, i2 * n2 + j1]
    
    print_matrix("A", A_true)
    print_matrix("Atilde", A_tilde_matrix)
    
    vec = np.random.randn(m2 * n2)
    print(f"Atilde matvec error: {norm(A_tilde_matrix @ vec - A_tilde.matvec(vec))}")
    vec = np.random.randn(m1 * n1)
    print(f"Atilde rmatvec error: {norm(A_tilde_matrix.T @ vec - A_tilde.rmatvec(vec))}")

    # Compute KSVD
    A_list_opt, B_list_opt, s_opt = kronecker_svd(A, m1, m2, n1, n2, r=r_true, method="lanczos")

    r_true = min(len(A_list_opt), r_true)
    # Reconstruct approximation
    A_approx_opt = sum(np.kron(A_list_opt[j], B_list_opt[j]) for j in range(r_true))

    # Compare
    error = np.linalg.norm(A_true - A_approx_opt)
    print("Reconstruction error SVD:", error)
    print("Singular values from reshaped SVD:", s_opt)
    
    vec = np.random.randn(m2 * n2)
    sol_exact = np.linalg.solve(A_true, vec)
    sol_precon_classical = np.linalg.solve(A_approx_opt, vec)
    err = norm(sol_exact - sol_precon_classical)
    print("Error of precoditioner: ", err)
    
    preconditioner = form_2d_preconditioner(A, m1, m2, n1, n2, r_true)
    A1 = preconditioner["A1"]
    A2 = preconditioner["A2"]
    B1 = preconditioner["B1"]
    B2 = preconditioner["B2"]

    P_approx = np.kron(A1, B1) + np.kron(A2, B2)

    x_sylv = apply_2d_preconditioner(vec, preconditioner, m2, n2)

    res = np.linalg.norm(P_approx @ x_sylv - vec)
    print("Residual ||P_approx x_sylv - b|| =", res)

# experiment 1
def make_snapshots():
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        show(lambda x, y: exact_solution(x, y, t), t) 

# experiment 2: explicit methods
def plot_explicit_errors(step_sizes, errors, initial_error, name="rk_convergence"):
    plt.figure(figsize=(8, 6))

    for rk in [1, 2, 3, 4]:
        plt.loglog(step_sizes, errors[rk], marker='o', label=f"RK{rk}")

    plt.gca().invert_xaxis()  # optional: decreasing h goes left->right
    plt.axhline(
        y=initial_error,
        color='k',
        linestyle='--',
        linewidth=1,
        label="initial precision"
    )
    plt.ylim(initial_error * 0.1, 1e1)
    plt.xlabel("Time step size h")
    plt.ylabel("Error at final time")
    plt.title("Runge--Kutta Convergence")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()

    plt.savefig(f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {name}.png")
    
def plot_explicit_runtimes(step_sizes, errors, name="rk_runtime"):
    plt.figure(figsize=(8, 6))

    for rk in [1, 2, 3, 4]:
        plt.loglog(step_sizes, errors[rk], marker='o', label=f"RK{rk}")

    plt.gca().invert_xaxis()
    plt.xlabel("Time step size h")
    plt.ylabel("Average iterations")
    plt.title("GMRES iterations")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()

    plt.savefig(f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {name}.png")
    
def run_explicit():
    explicit = False
    name = "a"
    name_time = "average_iterations_precon"
    t = start_time
    mesh = StructuredMesh2D(Nx, Ny)
    dof_handler = DoFHandler2D(mesh, Nq**2)
    cell_operator = CellWiseOperator(mesh)
    operator = AdvectionOperator(mesh, dof_handler, cell_operator, t)
    operator.assemble_system()
    U = operator.interpolate(t)
    initial_error = average(lambda x, y: operator.evaluate_function(U, x, y) - initial_solution(x, y))
    print(f"Average distance: {initial_error}")
            
    step_sizes = [0.01, 0.005]
    errors = {1: [], 2: [], 3: [], 4: []}
    runtime = {1: [], 2: [], 3: [], 4: []}
    for rk in [1, 2, 3, 4]:
        for h in step_sizes:
            print(f"Running experiment: rk={rk}, h={h}.")
            start = time.perf_counter()
            t = start_time
            operator.set_time(t)
            U = operator.interpolate(t)
            
            rk_A, rk_b, rk_c = rk_scheme(rk=rk, explicit=explicit)
            rk_stepper = RungeKuttaMethod(rk_A, rk_b, rk_c)
            def F(t, A):
                operator.set_time(t)
                return operator.apply(A)
            
            factor = 0
            if rk_stepper.A[0, 0] != 0:
                factor = rk_stepper.A[0, 0]
            elif rk > 1:
                factor = rk_stepper.A[1, 1]
            gmres_operator = operator.build_operator(h * factor)
            preconditioner = operator.build_preconditioner(h * factor)
            
            iter = 0 
            avg_iterations_global = 0
            while t < final_time:
                U, avg_iterations = rk_stepper.step(operator, F, A0=U , h=h, t=t, gmres_operator=gmres_operator, preconditioner=preconditioner)
                t += h
                avg_iterations_global += avg_iterations
                iter += 1
                err = average(lambda x, y: operator.evaluate_function(U, x, y) - exact_solution(x, y, t))
                print(f"Average error at time t={t}: {err}.")
                
            avg_iterations_global /= iter
            # compute average error at final time
            end = time.perf_counter()
            err = average(lambda x, y: operator.evaluate_function(U, x, y) - exact_solution(x, y, t))
            errors[rk].append(abs(err))
            # runtime[rk].append(end - start)
            runtime[rk].append(avg_iterations_global)
            print(avg_iterations_global)
    
    # plot_explicit_errors(step_sizes, errors, initial_error, name)
    plot_explicit_runtimes(step_sizes, runtime, name_time)

run_explicit()