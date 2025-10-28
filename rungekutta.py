import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator

# define rk schemes
def rk_scheme(rk, explicit = True):
    A = None
    b= None
    c = None
    if explicit == True:
        if rk == 4:
            A = [
                [0, 0, 0, 0],
                [1/2, 0, 0, 0],
                [0, 1/2, 0, 0],
                [0, 0, 1, 0]
            ]
            b = [1/6, 1/3, 1/3, 1/6]
            c = [0, 1/2, 1/2, 1]
        else:
            A = [[0]]
            b = [1]
            c = [0]
    else:
        if rk == 2:
            A = [
                [0, 0],
                [0.5, 0.5]
            ]
            b = [0, 0.5]
            c = [0, 1]
        else:
            A = [[1]]
            b = [1]
            c = [1]
    return A, b, c

class RungeKuttaMethod:
    """
    A general-purpose Runge-Kutta solver.
    
    This class is initialized with a Butcher tableau (A, b, c) and can
    perform time steps for any (autonomous) first-order ODE system 
    defined by dA/dt = F(A).
    
    The implementation currently supports EXPLICIT and DIRK methods only.
    """
    
    def __init__(self, A, b, c):
        """
        Initializes the solver with a given Butcher tableau.
        
        Args:
            A (list or np.array): The A matrix (s x s) of the tableau.
            b (list or np.array): The b vector (s,) of weights.
            c (list or np.array): The c vector (s,) of nodes.
        """
        self.A = np.array(A)
        self.b = np.array(b)
        self.c = np.array(c)
        
        # --- Property: self.stage ---
        num_stages = len(b)
        if (self.A.shape != (num_stages, num_stages) or 
            len(c) != num_stages):
            raise ValueError("Inconsistent dimensions for A, b, and c.")
        
        self.stage = num_stages
        
        # --- Property: self.explicit ---
        self.method_type = self._classify_method()

    def _classify_method(self):
        """
        Classifies the 'A' matrix as 'explicit', 'dirk', or 'implicit'.
        
        - explicit: Strictly lower-triangular (zeros on and above diagonal).
        - dirk: Lower-triangular (zeros strictly above diagonal).
        - implicit: Contains non-zero entries above the diagonal.
        """
        s = self.stage
        
        # 1. Check for explicit (zeros on and above diagonal)
        is_explicit = True
        for i in range(s):
            for j in range(i, s): # Start from j=i (the diagonal)
                if not np.isclose(self.A[i, j], 0):
                    is_explicit = False
                    break
            if not is_explicit:
                break
        
        if is_explicit:
            return "explicit"
            
        # 2. If not explicit, check for DIRK (zeros strictly above diagonal)
        is_lower_triangular = True
        for i in range(s):
            for j in range(i + 1, s): # Start from j=i+1 (above diagonal)
                if not np.isclose(self.A[i, j], 0):
                    is_lower_triangular = False
                    break
            if not is_lower_triangular:
                break
                
        if is_lower_triangular:
            return "dirk"
            
        # 3. If not explicit and not DIRK, it's fully implicit
        return "implicit"

    def step(self, operator, A0, h, t):
        """
        Performs a single time step of the Runge-Kutta method.
        
        This method solves dA/dt = F(A) using the formulas:
        Stage solution: A_j = A_0 + h * sum_l(a_jl * K_l)
        Stage derivative: K_j = F(A_j)
        Final solution:   A_1 = A_0 + h * sum_j(b_j * K_j)

        Args:
            F (callable): The right-hand side function F(A). It must take
                          a numpy array (current state) and return a
                          numpy array (the derivative).
            A0 (np.array or float): The solution at the current time (t_n).
            h (float): The step size.

        Returns:
            np.array or float: The solution at the new time (t_n+1).
        """
        
        if self.method_type == "explicit":
            A0 = np.asarray(A0)        
            K_stages = np.zeros((self.stage, *A0.shape), dtype=A0.dtype)

            for j in range(self.stage):
                # 1. Calculate the stage solution A_j
                if j == 0:
                    sum_term = np.zeros_like(A0)
                else:
                    sum_term = np.einsum('l,l...->...', self.A[j, :j], K_stages[:j])
                
                A_j = A0 + h * sum_term
                operator.set_time(t + h * self.c[j])
                K_stages[j] = operator.apply(A_j)

            # 3. Calculate the final solution A_1
            final_sum_term = np.einsum('j,j...->...', self.b, K_stages)
            A1 = A0 + h * final_sum_term
            
        elif self.method_type == "dirk":
            A0 = np.asarray(A0)        
            K_stages = np.zeros((self.stage, *A0.shape), dtype=A0.dtype)
            
            for j in range(self.stage):
                sum_term = operator.set_time(t + h * self.c[j])
                if j == 0 and self.A[0, 0] == 0: # first stage is explicit
                    K_stages[j] = operator.apply(A0)
                else:
                    rhs = A0 + h * np.einsum('l,l...->...', self.A[j, :j], K_stages[:j]) + h * self.A[j,j] * operator.Gbound
                    gmres_operator = LinearOperator(operator.M.shape, matvec=lambda v: v - h * self.A[j, j] * operator.M_inv @ (operator.B.T - operator.G) @ v)
                    
                    Yi, info = gmres(gmres_operator, rhs, M=None, tol=1e-10, maxiter=50) # preconditioner M
                    
                    K_stages[j] = operator.apply(Yi)
            
            final_sum_term = np.einsum('j,j...->...', self.b, K_stages)
            A1 = A0 + h * final_sum_term
        else:
            raise ValueError("General implicit methods are not implemented.")
        
        return A1
    