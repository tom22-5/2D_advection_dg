# Imports
import os
import time

import numpy as np
from numpy.linalg import norm, svd
from scipy.sparse import lil_matrix, coo_matrix
from scipy.sparse.linalg import LinearOperator
import matplotlib.pyplot as plt

from mesh import StructuredMesh2D
from dofhandler import DoFHandler2D
from legendre import legendre, legendre_derivative, gauss_legendre
from helpers import print_matrix, show, average, lagrange_basis, lagrange_basis_derivative, rsvd
from rungekutta import rk_scheme, RungeKuttaMethod
from lanczos import form_2d_preconditioner, apply_2d_preconditioner, apply_2d_preconditioner_transposed, lanczos_svd, kronecker_svd, apply_P1_forward, apply_P1_transposed_forward

# Hyperparameters
dim = 2
flux_alpha = 1.0
start_time = 0
final_time = 1.0
flux = "upwind" # upwind
velocity = "rotating" # constant, rotating, complicated
periodic = False # periodic, manufactured

# Hyperparameters grid
Nx = 16
Ny = 16
Lx = 1.0
Ly = 1.0
dx = Lx / Nx
dy = Ly / Ny

# Hyperparameters polynomials and quadrature
fe_degree = 4
Nq = fe_degree + 1
basis_type = "lagrange" # lagrange, legendre

print(f"Problem: linear advection, velocity={velocity}, flux={flux}, boundary={'periodic' if periodic else 'Dirichlet'}.")
print(f"Initializing DG solver with Nx={Nx}, Ny={Ny}, fe_degree={fe_degree}, basis={basis_type}.")

# exact solution at time = 0
def initial_solution(x1, x2):
    if velocity == "rotating":
        return exact_solution(x1, x2, 0)
    elif velocity == "complicated":
        return np.exp(-400. * ((x1 - 0.5) * (x1 - 0.5) + (x2 - 0.75) * (x2 - 0.75))) + np.exp(-100. * ((x1 - 0.5) * (x1 - 0.5) + (x2 - 0.25) * (x2 - 0.25)))
    elif velocity == "constant":
        return np.sin(2.0 * np.pi * x1) * np.sin(2.0 * np.pi * x2)
    else:
        raise NotImplementedError()

# exact analytical solution at any time
def exact_solution(x1, x2, t):
    if velocity == "rotating":
        return np.sin(np.pi * x1) * np.sin(np.pi * x2) * np.cos(t)
    elif velocity == "complicated":
        if np.abs(t) < 10e-08:
            return initial_solution(x1, x2)
        else:
            raise NotImplementedError()
    elif velocity == "constant":
        return initial_solution(x1 - transport_speed(x1, x2, t)[0] * t, x2 - transport_speed(x1, x2, t)[1] * t)
    else:
        raise NotImplementedError()
  
# exact solution at boundary
def boundary_solution(x1, x2, t):
    # if x1 != 1 and x1 != -1 and x2 != 1 and x2 != -1:
    #     raise ValueError("no boundary term")
    if velocity == "constant":
        return exact_solution(x1, x2, t)
    elif velocity == "rotating":
        return exact_solution(x1, x2, t)
    else:
        raise NotImplementedError()
  
# constant tranport speed in spacetime
def transport_speed(x1, x2, time = 0.0):
    if velocity == "rotating":
        return np.array([-x2, x1]) # rotating velocity vortex
    elif velocity == "complicated":
        factor = np.cos(np.pi * time / final_time) * 2.0
        
        pi_x1 = np.pi * x1
        pi_x2 = np.pi * x2
        osc_x = 2.0 * np.pi * (x1 + 0.2)
        osc_y = 2.0 * np.pi * (x2 + 0.3)
        
        sin_pi_x1 = np.sin(pi_x1)
        sin_pi_x2 = np.sin(pi_x2)
        
        sin_2pi_x1 = np.sin(2.0 * pi_x1)
        sin_2pi_x2 = np.sin(2.0 * pi_x2)
        
        sin_osc_x = np.sin(osc_x)
        cos_osc_x = np.cos(osc_x)
        
        sin_osc_y = np.sin(osc_y)
        cos_osc_y = np.cos(osc_y)
        
        u = factor * (sin_2pi_x2 * sin_pi_x1 * sin_pi_x1 + 0.2 * sin_osc_x * cos_osc_y)
        v = -factor * (sin_2pi_x1 * sin_pi_x2 * sin_pi_x2 + 0.2 * cos_osc_x * sin_osc_y)
        
        return np.array([u, v])
    elif velocity == "constant":
        return np.array([1.0, 1.0]) # constant advection
    else:
        raise NotImplementedError()
        
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

  def local_mass(self, ii, jj, tp = False, optimized = True):
    if tp:
        M_loc = np.kron(self.phi1D.T @ self.wts1D, self.phi1D.T @ self.wts1D) @ self.jac(ii, jj) @ np.kron(self.phi1D, self.phi1D)
    else:
        if optimized:
            det_jac = self.det_jac(ii, jj)
            w_scaled = np.array(self.quad_wts2D) * det_jac
            M_loc = self.phi.T @ (w_scaled[:, None] * self.phi)
        else:
            det_jac = self.det_jac(ii, jj)
            M_loc = np.zeros((Nq**2, Nq**2))
            for j in range(Nq**2):
                for i in range(Nq**2):
                    for q, w in enumerate(self.quad_wts2D):
                        M_loc[j,i] += w * self.phi[q,j] * self.phi[q,i] * det_jac
        
    return M_loc

  def local_volume(self, ii, jj, time = start_time, tp = False, optimized = True):
    if tp:
        raise NotImplementedError()
    else:           
        if optimized:
            det_jac = self.det_jac(ii, jj)
            jac = self.jac(ii, jj)
            Q = Nq**2
            V_q = np.zeros((Q, dim))
            for q, (xq, yq) in enumerate(self.quad_pts2D):
                x1_map, x2_map = self.map_to_original(ii, jj, xq, yq)
                V_q[q, :] = transport_speed(x1_map, x2_map, time)
            grad_phi_phys = np.einsum('ab, qjb -> qja', jac, self.dphi_dx)
            v_dot_grad = np.einsum('qa, qja -> qj', V_q, grad_phi_phys)
            w = np.array(self.quad_wts2D) * det_jac
            B_loc = np.einsum('q, qi, qj -> ji', w, self.phi, v_dot_grad)
        else:
            det_jac = self.det_jac(ii, jj)
            jac = self.jac(ii, jj)
            B_loc= np.zeros((Nq**2, Nq**2))
            for j in range(Nq**2):
                for i in range(Nq**2):
                    for q, w in enumerate(self.quad_wts2D):
                        x1 = self.quad_pts2D[q][0]
                        x2 = self.quad_pts2D[q][1]
                        x1_map, x2_map = self.map_to_original(ii, jj, x1, x2)
                        B_loc[j,i] += w * self.phi[q,i] * np.dot(transport_speed(x1_map, x2_map, time), jac @ self.dphi_dx[q,j,:]) * det_jac
    return B_loc

  def local_source(self, ii, jj, t):
    det_jac = self.det_jac(ii, jj)
    source_loc = np.zeros((Nq ** 2, 1))
    for i in range(Nq ** 2):
        for q, w in enumerate(self.quad_wts2D):
            x1 = self.quad_pts2D[q][0]
            x2 = self.quad_pts2D[q][1]
            x1_map, x2_map = self.map_to_original(ii, jj, x1, x2)
            
            if velocity == "rotating":
                term1 = -np.sin(np.pi * x1_map) * np.sin(np.pi * x2_map) * np.sin(t)
                term2 = x1_map * np.sin(np.pi * x1_map) * np.cos(np.pi * x2_map)
                term3 = -x2_map * np.cos(np.pi * x1_map) * np.sin(np.pi * x2_map)
                src_val = term1 + np.pi * np.cos(t) * (term2 + term3)
            else:
                src_val = 0.0
                
            source_loc[i] += w * self.phi[q,i] * src_val * det_jac
    return source_loc

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
  def __init__(self, mesh, dofhandler, cellop, t, optimized = True):
        self.mesh = mesh
        self.dofhandler = dofhandler
        self.cellop = cellop
        self.ndofs = dofhandler.ndofs
        self.time = t
        
        self.M = None
        self.M_inv = None
        self.B = None
        self.G = None
        self.B_minus_G = None
        if not optimized:
            self.M = lil_matrix((self.ndofs, self.ndofs))
            self.M_inv = lil_matrix((self.ndofs, self.ndofs))
            self.B = lil_matrix((self.ndofs, self.ndofs))
            self.G = lil_matrix((self.ndofs, self.ndofs))

        self.Gbound = np.zeros(self.ndofs)
        self.source = np.zeros(self.ndofs)

  def assemble_system(self, optimized = True):
      if optimized: 
        # 1. The Dump Trucks (Empty Python lists)
        M_row, M_col, M_data = [], [], []
        Minv_row, Minv_col, Minv_data = [], [], []
        B_row, B_col, B_data = [], [], []
        G_row, G_col, G_data = [], [], []
        
        # Ensure 1D arrays for vectors
        self.source = np.zeros(self.ndofs)
        self.Gbound = np.zeros(self.ndofs)

        # --- LOOP OVER CELLS ---
        for (ii, jj), dofs in self.dofhandler.iter_cells():
            # Fetch local matrices
            M_loc = self.cellop.local_mass(ii, jj)
            B_loc = self.cellop.local_volume(ii, jj, self.time)
            M_loc_inv = np.linalg.inv(M_loc)
            
            # Assign source vector instantly (DG cells don't share DoFs, so direct assignment is safe)
            self.source[dofs] = self.cellop.local_source(ii, jj, self.time).flatten()

            # Generate global index maps for this cell
            grid_i, grid_j = np.meshgrid(dofs, dofs, indexing='ij')
            
            # Shovel into dump trucks
            M_row.extend(grid_i.flatten())
            M_col.extend(grid_j.flatten())
            M_data.extend(M_loc.flatten())
            
            Minv_row.extend(grid_i.flatten())
            Minv_col.extend(grid_j.flatten())
            Minv_data.extend(M_loc_inv.flatten())
            
            B_row.extend(grid_i.flatten())
            B_col.extend(grid_j.flatten())
            B_data.extend(B_loc.flatten())

        # --- LOOP OVER FACES ---
        for face in self.dofhandler.iter_faces():
            dofs_cell = face["dofs_cell"]
            dofs_neighbor = face["dofs_neighbor"]
            normal = face["normal"]
            (ii, jj) = face["cell"]
            face_jac = self.cellop.det_jac_face(ii, jj, face["face"])
            
            if flux != "upwind":
                raise ValueError(f"Flux {flux} is not implemented.")
            
            if dofs_neighbor is None: # Boundary
                G_loc = np.zeros((len(dofs_cell), len(dofs_cell)))
                Gbound_loc = np.zeros(len(dofs_cell))
                
                for q, qpt in enumerate(self.cellop.qpts):
                    xq, yq = self.cellop.get_boundary_point(qpt, face["face"])
                    x1, x2 = self.cellop.map_to_original(ii, jj, xq, yq)
                    normal_speed = transport_speed(x1, x2, self.time) @ normal               
                    weight = normal_speed * self.cellop.qwts[q] * face_jac
                    
                    if normal_speed < 0: # inflow  
                        u_bnd = boundary_solution(x1, x2, self.time)
                        # Vectorized boundary addition
                        Gbound_loc += weight * self.cellop.bd_phi[face["face"]][q, :] * u_bnd
                    else: # outflow
                        # Vectorized outer product replaces the nested li, lj loop!
                        phi_j = self.cellop.bd_phi[face["face"]][q, :]
                        phi_i = self.cellop.bd_phi[face["face"]][q, :]
                        G_loc += weight * np.outer(phi_j, phi_i)
                        
                self.Gbound[dofs_cell] += Gbound_loc
                
                # Dump G_loc
                if np.any(G_loc):
                    grid_i, grid_j = np.meshgrid(dofs_cell, dofs_cell, indexing='ij')
                    G_row.extend(grid_i.flatten())
                    G_col.extend(grid_j.flatten())
                    G_data.extend(G_loc.flatten())
                                                
            else: # Interior face            
                G_loc_inflow = np.zeros((len(dofs_cell), len(dofs_neighbor)))
                G_loc_outflow = np.zeros((len(dofs_cell), len(dofs_cell)))
                
                for q, qpt in enumerate(self.cellop.qpts):
                    xq, yq = self.cellop.get_boundary_point(qpt, face["face"])
                    x1, x2 = self.cellop.map_to_original(ii, jj, xq, yq)
                    normal_speed = transport_speed(x1, x2, self.time) @ normal               
                    weight = normal_speed * self.cellop.qwts[q] * face_jac

                    if normal_speed < 0: # inflow  
                        face_neighbor = self.cellop.opposite(face["face"])
                        phi_j = self.cellop.bd_phi[face["face"]][q, :]
                        phi_i = self.cellop.bd_phi[face_neighbor][q, :]
                        G_loc_inflow += weight * np.outer(phi_j, phi_i)
                    else: # outflow
                        phi_j = self.cellop.bd_phi[face["face"]][q, :]
                        phi_i = self.cellop.bd_phi[face["face"]][q, :]
                        G_loc_outflow += weight * np.outer(phi_j, phi_i)
                        
                # Dump inflow
                if np.any(G_loc_inflow):
                    grid_i, grid_j = np.meshgrid(dofs_cell, dofs_neighbor, indexing='ij')
                    G_row.extend(grid_i.flatten())
                    G_col.extend(grid_j.flatten())
                    G_data.extend(G_loc_inflow.flatten())
                    
                # Dump outflow
                if np.any(G_loc_outflow):
                    grid_i, grid_j = np.meshgrid(dofs_cell, dofs_cell, indexing='ij')
                    G_row.extend(grid_i.flatten())
                    G_col.extend(grid_j.flatten())
                    G_data.extend(G_loc_outflow.flatten())
        
        # --- COMPILE SPARSE MATRICES ---
        # coo_matrix inherently sums duplicate (row, col) indices, which handles our flux overlaps perfectly!
        self.M = coo_matrix((M_data, (M_row, M_col)), shape=(self.ndofs, self.ndofs)).tocsr()
        self.M_inv = coo_matrix((Minv_data, (Minv_row, Minv_col)), shape=(self.ndofs, self.ndofs)).tocsr()
        self.B = coo_matrix((B_data, (B_row, B_col)), shape=(self.ndofs, self.ndofs)).tocsr()
        self.G = coo_matrix((G_data, (G_row, G_col)), shape=(self.ndofs, self.ndofs)).tocsr()
        self.B_minus_G = self.B - self.G
      else:
        # Loop over cells
        for (ii, jj), dofs in self.dofhandler.iter_cells():
            # local mass, volume and source
            M_loc = self.cellop.local_mass(ii, jj)
            B_loc = self.cellop.local_volume(ii, jj, self.time)
            M_loc_inv = np.linalg.inv(M_loc)
            source_loc = self.cellop.local_source(ii, jj, self.time)

            # global mass, volume and source
            for li, gi in enumerate(dofs):
                for lj, gj in enumerate(dofs):
                    self.M[gi, gj] += M_loc[li, lj]
                    self.B[gi, gj] += B_loc[li, lj]
                    self.M_inv[gi, gj] += M_loc_inv[li, lj]
                self.source[gi] = source_loc[li]

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
                        xq, yq = self.cellop.get_boundary_point(qpt, face["face"])
                        x1, x2 = self.cellop.map_to_original(ii, jj, xq, yq)
                        normal_speed = transport_speed(x1, x2, self.time) @ normal               
                        qwt = self.cellop.qwts[q]
                        
                        if normal_speed < 0: # inflow  
                            self.Gbound[gj] += normal_speed * qwt * face_jac * self.cellop.bd_phi[face["face"]][q, lj] * boundary_solution(x1, x2, self.time)
                        else: # outflow
                            for li, gi in enumerate(dofs_cell):
                                self.G[gj, gi] += normal_speed * qwt * face_jac * self.cellop.bd_phi[face["face"]][q, lj] * self.cellop.bd_phi[face["face"]][q, li]
                                                
            else: # interior face             
                for lj, gj in enumerate(dofs_cell):
                    for q, qpt in enumerate(self.cellop.qpts):
                        xq, yq = self.cellop.get_boundary_point(qpt, face["face"])
                        x1, x2 = self.cellop.map_to_original(ii, jj, xq, yq)
                        normal_speed = transport_speed(x1, x2, self.time) @ normal               
                        qwt = self.cellop.qwts[q]

                        if normal_speed < 0: # inflow  
                            for li, gi in enumerate(dofs_neighbor):
                                face_neighbor = self.cellop.opposite(face["face"])
                                self.G[gj, gi] += normal_speed * qwt * face_jac *  self.cellop.bd_phi[face["face"]][q, lj] * self.cellop.bd_phi[face_neighbor][q, li]
                        else: # outflow
                            for li, gi in enumerate(dofs_cell):
                                self.G[gj, gi] += normal_speed * qwt * face_jac *  self.cellop.bd_phi[face["face"]][q, lj] * self.cellop.bd_phi[face["face"]][q, li]
        
        self.M = self.M.tocsr()
        self.M_inv = self.M_inv.tocsr()
        self.B = self.B.tocsr()
        self.G = self.G.tocsr()
        self.B_minus_G = self.B - self.G
  
  def set_time(self, t, optimized = True):
      if optimized: 
        if self.time == t:
            return
        self.time = t
        
        self.Gbound = np.zeros(self.ndofs)
        self.source = np.zeros(self.ndofs)
        
        rebuild_matrices = (velocity == "complicated")
        
        B_row, B_col, B_data = [], [], []
        G_row, G_col, G_data = [], [], []

        # --- LOOP OVER CELLS ---
        for (ii, jj), dofs in self.dofhandler.iter_cells():
            self.source[dofs] = self.cellop.local_source(ii, jj, self.time).flatten()
                
            if rebuild_matrices:
                B_loc = self.cellop.local_volume(ii, jj, self.time)
                grid_i, grid_j = np.meshgrid(dofs, dofs, indexing='ij')
                B_row.extend(grid_i.flatten())
                B_col.extend(grid_j.flatten())
                B_data.extend(B_loc.flatten())

        # --- LOOP OVER FACES ---
        for face in self.dofhandler.iter_faces():
            dofs_cell = face["dofs_cell"]
            dofs_neighbor = face["dofs_neighbor"]
            normal = face["normal"]
            (ii, jj) = face["cell"]
            face_jac = self.cellop.det_jac_face(ii, jj, face["face"])
            
            if flux != "upwind":
                raise ValueError(f"Flux {flux} is not implemented.")
            
            if dofs_neighbor is None: # Boundary
                G_loc = np.zeros((len(dofs_cell), len(dofs_cell)))
                Gbound_loc = np.zeros(len(dofs_cell))
                
                for q, qpt in enumerate(self.cellop.qpts):
                    xq, yq = self.cellop.get_boundary_point(qpt, face["face"])
                    x1, x2 = self.cellop.map_to_original(ii, jj, xq, yq)
                    normal_speed = transport_speed(x1, x2, self.time) @ normal               
                    weight = normal_speed * self.cellop.qwts[q] * face_jac
                    
                    if normal_speed < 0: # inflow  
                        u_bnd = boundary_solution(x1, x2, self.time)
                        Gbound_loc += weight * self.cellop.bd_phi[face["face"]][q, :] * u_bnd
                    else: # outflow
                        if rebuild_matrices:
                            phi_j = self.cellop.bd_phi[face["face"]][q, :]
                            phi_i = self.cellop.bd_phi[face["face"]][q, :]
                            G_loc += weight * np.outer(phi_j, phi_i)
                            
                self.Gbound[dofs_cell] += Gbound_loc
                
                if rebuild_matrices and np.any(G_loc):
                    grid_i, grid_j = np.meshgrid(dofs_cell, dofs_cell, indexing='ij')
                    G_row.extend(grid_i.flatten())
                    G_col.extend(grid_j.flatten())
                    G_data.extend(G_loc.flatten())
                                                
            else: # Interior face            
                if rebuild_matrices:
                    G_loc_inflow = np.zeros((len(dofs_cell), len(dofs_neighbor)))
                    G_loc_outflow = np.zeros((len(dofs_cell), len(dofs_cell)))
                    
                    for q, qpt in enumerate(self.cellop.qpts):
                        xq, yq = self.cellop.get_boundary_point(qpt, face["face"])
                        x1, x2 = self.cellop.map_to_original(ii, jj, xq, yq)
                        normal_speed = transport_speed(x1, x2, self.time) @ normal               
                        weight = normal_speed * self.cellop.qwts[q] * face_jac

                        if normal_speed < 0: # inflow  
                            face_neighbor = self.cellop.opposite(face["face"])
                            phi_j = self.cellop.bd_phi[face["face"]][q, :]
                            phi_i = self.cellop.bd_phi[face_neighbor][q, :]
                            G_loc_inflow += weight * np.outer(phi_j, phi_i)
                        else: # outflow
                            phi_j = self.cellop.bd_phi[face["face"]][q, :]
                            phi_i = self.cellop.bd_phi[face["face"]][q, :]
                            G_loc_outflow += weight * np.outer(phi_j, phi_i)
                            
                    if np.any(G_loc_inflow):
                        grid_i, grid_j = np.meshgrid(dofs_cell, dofs_neighbor, indexing='ij')
                        G_row.extend(grid_i.flatten())
                        G_col.extend(grid_j.flatten())
                        G_data.extend(G_loc_inflow.flatten())
                        
                    if np.any(G_loc_outflow):
                        grid_i, grid_j = np.meshgrid(dofs_cell, dofs_cell, indexing='ij')
                        G_row.extend(grid_i.flatten())
                        G_col.extend(grid_j.flatten())
                        G_data.extend(G_loc_outflow.flatten())
        
        if rebuild_matrices:
            self.B = coo_matrix((B_data, (B_row, B_col)), shape=(self.ndofs, self.ndofs)).tocsr()
            self.G = coo_matrix((G_data, (G_row, G_col)), shape=(self.ndofs, self.ndofs)).tocsr()
            self.B_minus_G = self.B - self.G
      else:
        if self.time == t:
            return
        self.time = t
        
        self.Gbound = np.zeros(self.ndofs)
        self.source = np.zeros(self.ndofs)
        
        rebuild_matrices = (velocity == "complicated")
        
        if rebuild_matrices:
            self.B = lil_matrix((self.ndofs, self.ndofs))
            self.G = lil_matrix((self.ndofs, self.ndofs))

        for (ii, jj), dofs in self.dofhandler.iter_cells():
            source_loc = self.cellop.local_source(ii, jj, self.time)
            for li, gi in enumerate(dofs):
                self.source[gi] = source_loc[li]
                
            # Only rebuild the volume matrix if velocity is changing
            if rebuild_matrices:
                B_loc = self.cellop.local_volume(ii, jj, self.time)
                for li, gi in enumerate(dofs):
                    for lj, gj in enumerate(dofs):
                        self.B[gi, gj] += B_loc[li, lj]

        for face in self.dofhandler.iter_faces():
            dofs_cell = face["dofs_cell"]
            dofs_neighbor = face["dofs_neighbor"]
            normal = face["normal"]
            (ii, jj) = face["cell"]
            face_jac = self.cellop.det_jac_face(ii, jj, face["face"])
            
            if flux != "upwind":
                raise ValueError(f"Flux {flux} is not implemented.")
            
            if dofs_neighbor is None: # Physical Boundary
                for lj, gj in enumerate(dofs_cell):
                    for q, qpt in enumerate(self.cellop.qpts):
                        xq, yq = self.cellop.get_boundary_point(qpt, face["face"])
                        x1, x2 = self.cellop.map_to_original(ii, jj, xq, yq)
                        normal_speed = transport_speed(x1, x2, self.time) @ normal               
                        qwt = self.cellop.qwts[q]
                        
                        if normal_speed < 0: # inflow  
                            self.Gbound[gj] += normal_speed * qwt * face_jac * self.cellop.bd_phi[face["face"]][q, lj] * boundary_solution(x1, x2, self.time)
                        else:
                            if rebuild_matrices:
                                for li, gi in enumerate(dofs_cell):
                                    self.G[gj, gi] += normal_speed * qwt * face_jac * self.cellop.bd_phi[face["face"]][q, lj] * self.cellop.bd_phi[face["face"]][q, li]
                                                
            else: # Interior face            
                if rebuild_matrices:
                    for lj, gj in enumerate(dofs_cell):
                        for q, qpt in enumerate(self.cellop.qpts):
                            xq, yq = self.cellop.get_boundary_point(qpt, face["face"])
                            x1, x2 = self.cellop.map_to_original(ii, jj, xq, yq)
                            normal_speed = transport_speed(x1, x2, self.time) @ normal               
                            qwt = self.cellop.qwts[q]

                            if normal_speed < 0: # inflow  
                                for li, gi in enumerate(dofs_neighbor):
                                    face_neighbor = self.cellop.opposite(face["face"])
                                    self.G[gj, gi] += normal_speed * qwt * face_jac * self.cellop.bd_phi[face["face"]][q, lj] * self.cellop.bd_phi[face_neighbor][q, li]
                            else: # outflow
                                for li, gi in enumerate(dofs_cell):
                                    self.G[gj, gi] += normal_speed * qwt * face_jac * self.cellop.bd_phi[face["face"]][q, lj] * self.cellop.bd_phi[face["face"]][q, li]
        
        if rebuild_matrices:
            self.B = self.B.tocsr()
            self.G = self.G.tocsr()
            self.B_minus_G = self.B - self.G
            
  def apply(self, src, optimized = True):
    return self.M_inv @ (self.B_minus_G @ src - self.Gbound + self.source)
  
  def apply_volume(self, src):
      return self.B @ src
  
  def apply_boundary(self, src):
      return self.G @ src
  
  def add_boundary(self):
      return self.Gbound
  
  def interpolate(self, t, optimized = True):
      F = np.zeros(self.ndofs)
      for (i, j), dofs in self.dofhandler.iter_cells():
          # local F
          F_loc = self.cellop.local_f(i, j, t)
          #   print_matrix(f"F_loc({i},{j})", F_loc)
          
          # global mass and volume
          for li, gi in enumerate(dofs):
            F[gi] += F_loc[li]
      U = self.M_inv @ F
      if optimized:
        return U.flatten()
      else:
        return U
  
  def evaluate_function(self, U, x, y):
    for (i, j), dofs in self.dofhandler.iter_cells():
        x0, x1, y0, y1 = self.mesh.cell_bounds(i, j)
        if x0 <= x <= x1 and y0 <= y <= y1:
            z1, z2 = self.cellop.map_to_reference(i, j, x, y)
            return self.cellop.evaluate_function(U, z1, z2, dofs)

    raise ValueError("point outside domain")
  
  def build_operator(self, factor, optimized = True):
    n = self.ndofs
    if optimized:
      def matvec(v_flat):
          v = v_flat.reshape(n) # keep it 1D
          w = v - factor * (self.M_inv @ (self.B_minus_G @ v))
          return w
      
      def rmatvec(v_flat):
          v = v_flat.reshape(n)
          w = v - factor * (self.B_minus_G.T @ (self.M_inv.T @ v))
          return w
          
      return LinearOperator(shape=(n, n), matvec=matvec, rmatvec=rmatvec, dtype=float)
    else:
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
  
  def build_local_operator(self, dofs, factor):
    dofs = np.array(dofs)
    Mloc = self.M_inv[np.ix_(dofs, dofs)].toarray()
    Bloc = self.B    [np.ix_(dofs, dofs)].toarray()
    Gloc = self.G    [np.ix_(dofs, dofs)].toarray() 
    # Mloc = self.M_inv[np.ix_(dofs, dofs)]
    # Bloc = self.B    [np.ix_(dofs, dofs)]
    # Gloc = self.G    [np.ix_(dofs, dofs)]
    
    # Local operators
    def loc_matvec(v_loc):
        return v_loc - factor * (Mloc @ ((Bloc - Gloc) @ v_loc))
        
    def loc_tmatvec(v_loc):
        return v_loc - factor * ((Bloc.T - Gloc.T) @ (Mloc.T @ v_loc))
                
    local_operator = LinearOperator(
        shape=(Nq**2, Nq**2),
        matvec=loc_matvec,
        rmatvec=loc_tmatvec,
        dtype=float
    )
    return local_operator
  
  def build_local_preconditioner(self, local_operator, type = "pazner", r=None, k=None):
    if type == "pazner":
        precon, _ = self.build_local_pazner(local_operator, r)
        return precon
    elif type == "pazner_woodbury":
        return self.build_local_woodbury(local_operator, k, "pazner", r)
    elif type == "diagonal_woodbury":
        return self.build_local_woodbury(local_operator, k, "diagonal", None)
    elif type == "FDM":
        raise NotImplementedError()
    elif type == "ADI":
        raise NotImplementedError()
    elif type == "diagonal":
        precon, _ = self.build_local_diagonal(local_operator)
        return precon
    else:
        raise NotImplementedError()
    
  def build_local_pazner(self, local_operator, r):
    precon_dict = form_2d_preconditioner(local_operator, Nq, Nq, Nq, Nq, r)
    
    def loc_matvec(v):
        return apply_2d_preconditioner(v, precon_dict, m2=None, n2=None, r=r)
    
    preconditioner = LinearOperator(
        shape=(Nq**2, Nq**2),
        matvec=loc_matvec,
        dtype=float
    )
    
    def loc_inv_matvec(v_loc):
        return apply_P1_forward(v_loc, precon_dict, r)
    
    def loc_inv_tmatvec(v_loc):
        return apply_P1_transposed_forward(v_loc, precon_dict, r)
    
    inv_preconditioner = LinearOperator(
        shape=(Nq**2, Nq**2),
        matvec=loc_inv_matvec,
        rmatvec=loc_inv_tmatvec,
        dtype=float
    )
    
    return preconditioner, inv_preconditioner

  def build_local_woodbury(self, local_operator, k, type = "pazner", r=None):
    if type == "pazner":
        base_precon, base_inv_precon = self.build_local_pazner(local_operator, r)
    elif type == "diagonal":
        base_precon, base_inv_precon = self.build_local_diagonal(local_operator)
    else:
        raise NotImplementedError()
      
    def loc_rest_matvec(v_loc):
        return local_operator.matmat(v_loc) - base_inv_precon.matmat(v_loc)
    
    def loc_rest_tmatvec(v_loc):
        return local_operator.rmatmat(v_loc) - base_inv_precon.rmatmat(v_loc)
    
    # Randomized SVD (Note: pass ndofs_loc, not self.ndofs!)
    U, V = rsvd(loc_rest_matvec, loc_rest_tmatvec, Nq**2, k, factors=2)
    
    # Precompute PU = P1^{-1} U (Process column-by-column)
    PU = np.zeros_like(U)
    for col in range(k):
        PU[:, col] = base_precon.matvec(U[:, col])
        
    # Precompute Woodbury Core = (I_k + V^T P1^{-1} U)^{-1}
    core_inv = np.linalg.pinv(np.eye(k) + V @ PU, rcond=1e-12)
    
    def loc_matvec(v_local):
        w_base = base_precon.matvec(v_local)        
        update = PU @ (core_inv @ (V @ w_base)) # Woodbury update: PU @ core_inv @ V^T @ w_base
        return w_base - update
    
    return LinearOperator(
        shape=(Nq**2, Nq**2),
        matvec=loc_matvec,
        dtype=float
    )
  
  def build_local_diagonal(self, local_operator):
      N = local_operator.shape[0]
      diag_A = np.zeros(N)
      
      # 1. Extract the diagonal using standard basis vectors
      e = np.zeros(N)
      for i in range(N):
          e[i] = 1.0
          diag_A[i] = local_operator.matvec(e)[i]
          e[i] = 0.0
          
      # 2. The Preconditioner (Inverse Diagonal)
      inv_diag_A = 1.0 / diag_A
      
      def loc_matvec(v_loc):
          # Diagonal application is just element-wise multiplication!
          return inv_diag_A * v_loc.flatten()
          
      def loc_tmatvec(v_loc):
          return inv_diag_A * v_loc.flatten()
          
      preconditioner = LinearOperator(
          shape=(N, N),
          matvec=loc_matvec,
          rmatvec=loc_tmatvec,
          dtype=float
      )
      
      # 3. The Forward Operator (Required for Woodbury error sketching E = A - P)
      def loc_inv_matvec(v_loc):
          return diag_A * v_loc.flatten()
          
      def loc_inv_tmatvec(v_loc):
          return diag_A * v_loc.flatten()
          
      inv_preconditioner = LinearOperator(
          shape=(N, N),
          matvec=loc_inv_matvec,
          rmatvec=loc_inv_tmatvec,
          dtype=float
      )
      
      return preconditioner, inv_preconditioner
  
  def build_local_fdm(self, local_operator):
      pass
  
  def build_local_adi(self, local_operator):
      pass
  
  def build_preconditioner(self, factor, type = "pazner", r=None, k=None):        
    # --- 1. SETUP PHASE (Runs ONCE) ---
    cell_preconditioners = {}
    
    for (ii, jj), dofs in self.dofhandler.iter_cells():
        local_operator = self.build_local_operator(dofs, factor)
        cell_preconditioners[(ii, jj)] = self.build_local_preconditioner(local_operator, type, r, k)

    # --- 2. APPLICATION PHASE (Runs EVERY GMRES iteration) ---
    def matvec(v):
        v = v.reshape(self.ndofs)
        w = np.zeros(self.ndofs)
        
        for (ii, jj), dofs in self.dofhandler.iter_cells():
            v_local = v[dofs]
            w[dofs] = cell_preconditioners[(ii, jj)].matvec(v_local)
        
        return w

    return LinearOperator(shape=(self.ndofs, self.ndofs), matvec=matvec, dtype=float)
      
### tests
# test 1: constant solution
# only works for constant velocity
def test_constant_solution():
    t = 0.0
    mesh = StructuredMesh2D(Nx, Ny, periodic)
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
    # for (ii, jj), dofs in op.dofhandler.iter_cells():
    #     cell_vals = ge[dofs].ravel()
    #     print(f"cell ({ii},{jj})  G e =", cell_vals)
        
    # print(f"Global Gbound: {op.Gbound}")

# test 2: time derivative for 2D sine
def test_time_derivative():
    t = 0.25
    mesh = StructuredMesh2D(Nx, Ny, periodic)
    dof_handler = DoFHandler2D(mesh, Nq**2)
    cell_operator = CellWiseOperator(mesh)
    op = AdvectionOperator(mesh, dof_handler, cell_operator, t)
    op.assemble_system()
    Ut_exact_proj = project_exact_ut(op, t)
    U = op.interpolate(t)

    op.set_time(t)
    RHS = op.apply(U)

    diff = RHS - Ut_exact_proj
    print("||RHS||           =", norm(RHS))
    print("||Ut_exact_proj|| =", norm(Ut_exact_proj))
    print("||diff||          =", norm(diff))
    print("Relative error    =", norm(diff)/norm(Ut_exact_proj))
    
def exact_ut(x, y, t):
    if velocity == "rotating":
        return -np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(t)
    elif velocity == "constant":
        return -(1.0 * (2*np.pi * np.cos(2*np.pi*x) * np.sin(2*np.pi*y))
                 + 1.0 * (2*np.pi * np.sin(2*np.pi*x) * np.cos(2*np.pi*y)))
    else:
        raise NotImplementedError()

def project_exact_ut(operator, t):
    F = np.zeros((operator.ndofs, 1))
    for (ii, jj), dofs in operator.dofhandler.iter_cells():
        det_jac = operator.cellop.det_jac(ii, jj)
        F_loc = np.zeros((Nq**2, 1))
        for i in range(Nq**2):
            for q, w in enumerate(operator.cellop.quad_wts2D):
                z1, z2 = operator.cellop.quad_pts2D[q]
                x, y = operator.cellop.map_to_original(ii, jj, z1, z2)
                F_loc[i] += w * operator.cellop.phi[q, i] * exact_ut(x, y, t) * det_jac
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

### experiments
def make_snapshots():
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        print(f"Printing solution at t = {t}.")
        show(lambda x, y: exact_solution(x, y, t), t, f"-{velocity}") 

def plot_errors(step_sizes, errors, initial_error, name="rk_convergence"):
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

    save_dir = "./plots"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {name}.png")
    
def plot_runtimes(step_sizes, errors, name="rk_runtime"):
    plt.figure(figsize=(8, 6))

    for rk in [1, 2, 3, 4]:
        plt.loglog(step_sizes, errors[rk], marker='o', label=f"RK{rk}")

    plt.gca().invert_xaxis()
    plt.xlabel("Time step size h")
    plt.ylabel("Runtime")
    plt.title("Runge--Kutta Runtime")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()

    save_dir = "./plots"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {name}.png")
    
def plot_iterations(step_sizes, errors, name="rk_iterations"):
    plt.figure(figsize=(8, 6))

    for rk in [1, 2, 3, 4]:
        plt.loglog(step_sizes, errors[rk], marker='o', label=f"RK{rk}")

    plt.gca().invert_xaxis()
    plt.xlabel("Time step size h")
    plt.ylabel("Average iterations")
    plt.title("GMRES iterations")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()

    save_dir = "./plots"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {name}.png")
   
def run_explicit():
    explicit = True
    name = f"{velocity}_rk_explicit_convergence"
    name_time = f"{velocity}_rk_explicit_runtime"
    t = start_time
    mesh = StructuredMesh2D(Nx, Ny, periodic)
    dof_handler = DoFHandler2D(mesh, Nq**2)
    cell_operator = CellWiseOperator(mesh)
    operator = AdvectionOperator(mesh, dof_handler, cell_operator, t)
    operator.assemble_system()
    U = operator.interpolate(t)
    initial_error = average(lambda x, y: operator.evaluate_function(U, x, y) - initial_solution(x, y))
    print(f"Average distance: {initial_error}")
            
    step_sizes = [0.1, 0.05, 0.01, 0.005]
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
            
            iter = 0 
            while np.abs(t - final_time) > h / 2:
                U, _ = rk_stepper.step(operator, F, A0=U , h=h, t=t, gmres_operator=None, preconditioner=None)
                t += h
                iter += 1
                err = average(lambda x, y: operator.evaluate_function(U, x, y) - exact_solution(x, y, t))
                print(f"Average error at time t={t}: {err}.")
                
            end = time.perf_counter()
            err = average(lambda x, y: operator.evaluate_function(U, x, y) - exact_solution(x, y, t))
            errors[rk].append(abs(err))
            runtime[rk].append(end - start)
    
    plot_errors(step_sizes, errors, initial_error, name)
    print(errors)
    plot_runtimes(step_sizes, runtime, name_time)
    print(runtime)

def run_implicit():
    explicit = False
    name = f"{velocity}_rk_implicit_convergence"
    name_time = f"{velocity}_rk_implicit_runtime"
    name_iterations = f"{velocity}_rk_implicit_iterations"
    t = start_time
    mesh = StructuredMesh2D(Nx, Ny, periodic)
    dof_handler = DoFHandler2D(mesh, Nq**2)
    cell_operator = CellWiseOperator(mesh)
    operator = AdvectionOperator(mesh, dof_handler, cell_operator, t)
    operator.assemble_system()
    U = operator.interpolate(t)
    initial_error = average(lambda x, y: operator.evaluate_function(U, x, y) - initial_solution(x, y))
    print(f"Average distance: {initial_error}")
            
    step_sizes = [0.05]
    errors = {1: [], 2: [], 3: [], 4: []}
    runtime = {1: [], 2: [], 3: [], 4: []}
    iterations = {1: [], 2: [], 3: [], 4: []}
    for rk in [4]:
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
            
            iter = 0 
            avg_iterations_global = 0
            while np.abs(t - final_time) > h / 2:
                U, avg_iterations = rk_stepper.step(operator, F, A0=U , h=h, t=t, gmres_operator=gmres_operator, preconditioner=None)
                t += h
                avg_iterations_global += avg_iterations
                iter += 1
                err = average(lambda x, y: operator.evaluate_function(U, x, y) - exact_solution(x, y, t))
                print(f"Average error at time t={t}: {err}.")
                
            avg_iterations_global /= iter
            end = time.perf_counter()
            err = average(lambda x, y: operator.evaluate_function(U, x, y) - exact_solution(x, y, t))
            errors[rk].append(abs(err))
            runtime[rk].append(end - start)
            iterations[rk].append(avg_iterations_global)
            # print(avg_iterations_global)
    
    plot_errors(step_sizes, errors, initial_error, name)
    print(errors)
    plot_runtimes(step_sizes, runtime, name_time)
    print(runtime)
    plot_iterations(step_sizes, iterations, name_iterations)
    print(iterations)

def run_precon():
    explicit = False
    name = f"{velocity}_rk_precon_convergence"
    name_time = f"{velocity}_rk_precon_runtime"
    name_iterations = f"{velocity}_rk_precon_iterations"
    t = start_time
    mesh = StructuredMesh2D(Nx, Ny, periodic)
    dof_handler = DoFHandler2D(mesh, Nq**2)
    cell_operator = CellWiseOperator(mesh)
    operator = AdvectionOperator(mesh, dof_handler, cell_operator, t)
    operator.assemble_system()
    U = operator.interpolate(t)
    initial_error = average(lambda x, y: operator.evaluate_function(U, x, y) - initial_solution(x, y))
    print(f"Average distance: {initial_error}")
            
    step_sizes = [0.1, 0.05, 0.01, 0.005]
    errors = {1: [], 2: [], 3: [], 4: []}
    runtime = {1: [], 2: [], 3: [], 4: []}
    iterations = {1: [], 2: [], 3: [], 4: []}
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
            preconditioner = operator.build_preconditioner(h * factor, "pazner", r=1)
            
            iter = 0 
            avg_iterations_global = 0
            while np.abs(t - final_time) > h / 2:
                U, avg_iterations = rk_stepper.step(operator, F, A0=U , h=h, t=t, gmres_operator=gmres_operator, preconditioner=preconditioner)
                t += h
                avg_iterations_global += avg_iterations
                iter += 1
                # err = average(lambda x, y: operator.evaluate_function(U, x, y) - exact_solution(x, y, t))
                # print(f"Average error at time t={t}: {err}.")
                
            avg_iterations_global /= iter
            end = time.perf_counter()
            err = average(lambda x, y: operator.evaluate_function(U, x, y) - exact_solution(x, y, t))
            errors[rk].append(abs(err))
            runtime[rk].append(end - start)
            iterations[rk].append(avg_iterations_global)
            print(avg_iterations_global)
    
    plot_errors(step_sizes, errors, initial_error, name)
    print(errors)
    plot_runtimes(step_sizes, runtime, name_time)
    print(runtime)
    plot_iterations(step_sizes, iterations, name_iterations)
    print(iterations)

def run_improved_precon():
    explicit = False
    t = start_time
    mesh = StructuredMesh2D(Nx, Ny, periodic)
    dof_handler = DoFHandler2D(mesh, Nq**2)
    cell_operator = CellWiseOperator(mesh)
    operator = AdvectionOperator(mesh, dof_handler, cell_operator, t)
    operator.assemble_system()
    U = operator.interpolate(t)
    initial_error = average(lambda x, y: operator.evaluate_function(U, x, y) - initial_solution(x, y))
    print(f"Average distance: {initial_error}")
            
    step_sizes = [0.1, 0.05, 0.01, 0.005]
    ranks = [2, 4, 8, 16]
    types = ["pazner2", "pazner2_woodbury"]
    for type in types:
        for k in ranks:
            if type in ["pazner1", "pazner2", "diagonal"] and k > 2:
                continue
            errors = {1: [], 2: [], 3: [], 4: []}
            runtime = {1: [], 2: [], 3: [], 4: []}
            iterations = {1: [], 2: [], 3: [], 4: []}
            name = f"{type}_{velocity}_rk_improved_precon_convergence_k={k}"
            name_time = f"{type}_{velocity}_rk_improved_precon_runtime_k={k}"
            name_iterations = f"{type}_{velocity}_rk_improved_precon_iterations_k={k}"
            
            print(f"Running experiment: {type}, k={k}.")
            for rk in [1, 2, 3, 4]:
                for h in step_sizes:
                    print(f"Running experiment: {type}, rk={rk}, h={h}, k={k}.")
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
                    preconditioner = LinearOperator(shape=(operator.ndofs, operator.ndofs), matvec=lambda v: v,dtype=float)
                    if type == "pazner1":
                        preconditioner = operator.build_preconditioner(h * factor, "pazner", r=1, k=k)
                    elif type == "pazner2":
                        preconditioner = operator.build_preconditioner(h * factor, "pazner", r=2, k=k)
                    elif type == "pazner1_woodbury":
                        preconditioner = operator.build_preconditioner(h * factor, "pazner_woodbury", r=1, k=k)
                    elif type == "pazner2_woodbury":
                        preconditioner = operator.build_preconditioner(h * factor, "pazner_woodbury", r=2, k=k)
                    else:
                        preconditioner = operator.build_preconditioner(h * factor, type, r=None, k=k)
                    
                    iter = 0 
                    avg_iterations_global = 0
                    while np.abs(t - final_time) > h / 2:
                        U, avg_iterations = rk_stepper.step(operator, F, A0=U , h=h, t=t, gmres_operator=gmres_operator, preconditioner=preconditioner)
                        t += h
                        avg_iterations_global += avg_iterations
                        iter += 1
                        # err = average(lambda x, y: operator.evaluate_function(U, x, y) - exact_solution(x, y, t))
                        # print(f"Average error at time t={t}: {err}.")
                        
                    avg_iterations_global /= iter
                    end = time.perf_counter()
                    # err = average(lambda x, y: operator.evaluate_function(U, x, y) - exact_solution(x, y, t))
                    # errors[rk].append(abs(err))
                    runtime[rk].append(end - start)
                    iterations[rk].append(avg_iterations_global)
                    print(avg_iterations_global)
    
            # plot_errors(step_sizes, errors, initial_error, name)
            # print(errors)
            plot_runtimes(step_sizes, runtime, name_time)
            print(runtime)
            plot_iterations(step_sizes, iterations, name_iterations)
            print(iterations)

def run_one():
    t = start_time
    mesh = StructuredMesh2D(Nx, Ny, periodic)
    dof_handler = DoFHandler2D(mesh, Nq**2)
    cell_operator = CellWiseOperator(mesh)
    operator = AdvectionOperator(mesh, dof_handler, cell_operator, t)
    operator.assemble_system()
    U = operator.interpolate(t)
    
    initial_error = average(lambda x, y: operator.evaluate_function(U, x, y) - initial_solution(x, y))
    print(f"Average distance: {initial_error}")
    # print(f"Printing solution at t = {t}.")
    # show(lambda x, y: operator.evaluate_function(U, x, y), t, f"-{velocity}")
            
    h = 0.005
    k = 0
    type = "pazner2"        
    print(f"Running experiment: {type}, k={k}.")
    start = time.perf_counter()
    t = start_time
    
    rk = 4
    rk_A, rk_b, rk_c = rk_scheme(rk=rk, explicit=False)
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
    preconditioner = LinearOperator(shape=(operator.ndofs, operator.ndofs), matvec=lambda v: v,dtype=float)
    if type == "pazner1":
        preconditioner = operator.build_preconditioner(h * factor, "pazner", r=1, k=k)
    elif type == "pazner2":
        preconditioner = operator.build_preconditioner(h * factor, "pazner", r=2, k=k)
    elif type == "pazner1_woodbury":
        preconditioner = operator.build_preconditioner(h * factor, "pazner_woodbury", r=1, k=k)
    elif type == "pazner2_woodbury":
        preconditioner = operator.build_preconditioner(h * factor, "pazner_woodbury", r=2, k=k)
    else:
        preconditioner = operator.build_preconditioner(h * factor, type, r=None, k=k)
    
    iter = 0 
    avg_iterations_global = 0
    snaptimes = [0.1 * k * final_time for k in range(11)]
    while np.abs(t - final_time) > h / 2:
        U, avg_iterations = rk_stepper.step(operator, F, A0=U , h=h, t=t, gmres_operator=gmres_operator, preconditioner=preconditioner)
        t += h
        avg_iterations_global += avg_iterations
        iter += 1
        err = average(lambda x, y: operator.evaluate_function(U, x, y) - exact_solution(x, y, t))
        print(f"Average error at time t={t}: {err}.")
        print(f"Time = {t}")
        for snaptime in snaptimes:
            if np.abs(t - snaptime) < h / 2:
                print(f"Printing solution at t = {t}.")
                show(lambda x, y: operator.evaluate_function(U, x, y), t, f"-{velocity}") 
        
        
    avg_iterations_global /= iter
    end = time.perf_counter()
    err = average(lambda x, y: operator.evaluate_function(U, x, y) - exact_solution(x, y, t))
    print(avg_iterations_global)
    print(f"Average error at time t={t}: {err}.")
    print(f"Runtime: {end - start} s")
    
# test setup
# test_constant_solution()
# test_time_derivative()

# run experiments
run_one()
# run_explicit()
# run_implicit()
# run_improved_precon()

# Furkan.Akin@