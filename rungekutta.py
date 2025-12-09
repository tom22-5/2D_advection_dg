import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
from lanczos import form_2d_preconditioner, apply_2d_preconditioner

# define rk schemes
def rk_scheme(rk, explicit = True):
    A = None
    b= None
    c = None
    if explicit == True:
        if rk == 4: # Classical explicit RK4 (order 4)
            A = [
                [0, 0, 0, 0],
                [1/2, 0, 0, 0],
                [0, 1/2, 0, 0],
                [0, 0, 1, 0]
            ]
            b = [1/6, 1/3, 1/3, 1/6]
            c = [0, 1/2, 1/2, 1]
        elif rk == 3:  # Classical explicit RK3 (order 3)
            A = [
                [0,   0, 0],
                [1/2, 0, 0],
                [-1,  2, 0]
            ]
            b = [1/6, 2/3, 1/6]
            c = [0, 1/2, 1]
        elif rk == 2:  # Heun (explicit RK2)
            A = [
                [0, 0],
                [1, 0]
            ]
            b = [1/2, 1/2]
            c = [0, 1]
        else: # Euler
            A = [[0]]
            b = [1]
            c = [0]
    else:
        if rk == 4:
            gamma = 0.435866521508

            A = [
                [0.0,             0.0,             0.0,            0.0],
                [gamma,           gamma,           0.0,            0.0],
                [
                    (-4 * np.square(gamma) + 6 * gamma -1) / (4 * gamma),
                    (-2 * gamma + 1) / (4 * gamma), 
                    gamma,
                    0.0
                ],
                [
                    (6 * gamma - 1) / (12 * gamma),    
                    (-1) / ((24 * gamma - 12) * gamma),
                    (-6 * np.square(gamma) + 6 * gamma - 1) / (6 * gamma - 3), 
                    gamma
                ]
            ]

            b = [
                (6 * gamma - 1) / (12 * gamma),    
                (-1) / ((24 * gamma - 12) * gamma),
                (-6 * np.square(gamma) + 6 * gamma - 1) / (6 * gamma - 3), 
                gamma
            ]

            c = [0.0,
                2*gamma,
                1.0,
                1.0]
        elif rk == 3: # SDIRK-NC34
            gamma = (3 + 2 * np.sqrt(3) * np.cos(np.pi / 18)) / 6
            A = [
                [gamma,       0.0,       0.0],
                [1/2 - gamma,       gamma,       0.0],
                [2 * gamma,     1 - 4 * gamma,     gamma]
            ]

            b = [
                1 / (6 * np.square(1 - 2 * gamma)), 
                (2 * (1 - 6 * gamma + 6 * np.square(gamma))) / (3 * np.square(2 * gamma - 1)),
                1 / (6 * np.square(1 - 2 * gamma))
            ]

            c = [
                gamma,
                1 / 2,
                1 - gamma   
            ]
        elif rk == 2: # SDIRK-NCS23
            gamma = (3 + np.sqrt(3)) / 6
            A = [
                [gamma, 0],
                [1 - 2 * gamma , gamma]
            ]
            b = [1/2, 1/2]
            c = [gamma, 1 - gamma]
        else: # Implicit Euler
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

    def step(self, operator, F, A0, h, t, gmres_operator=None, preconditioner=None):
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
                # compute sum_term = sum_{l=0}^{j-1} A[j,l] * K_stages[l]
                sum_term = np.zeros_like(A0)
                for l in range(j):  # only sum previous stages
                    sum_term += self.A[j, l] * K_stages[l]

                # stage solution
                A_j = A0 + h * sum_term

                K_stages[j] = F(t + h * self.c[j], A_j)

            # final solution: A1 = A0 + h * sum_j b[j] * K_stages[j]
            final_sum_term = np.zeros_like(A0)
            for j in range(self.stage):
                final_sum_term += self.b[j] * K_stages[j]

            A1 = A0 + h * final_sum_term
            
        elif self.method_type == "dirk":
            A0 = np.asarray(A0)       
            K_stages = np.zeros((self.stage, *A0.shape), dtype=A0.dtype)

            for j in range(self.stage):
                # Case: first stage explicit (A[0,0] = 0)
                operator.set_time(t + h * self.c[j])
                if j == 0 and self.A[0, 0] == 0:
                    K_stages[j] = operator.M_inv @ ((operator.B - operator.G) @ A0 - operator.Gbound)
                else:
                    # Compute sum_{l=0}^{j-1} A[j,l] * K_stages[l]
                    sum_prev = np.zeros_like(A0)
                    for l in range(j):  # only previous stages
                        sum_prev += self.A[j, l] * K_stages[l]

                    # rhs = A0 + h * sum_{l<j} A[j,l] K_l + h * A[j,j] * Gbound
                    rhs = A0 + h * sum_prev - h * self.A[j, j] * operator.M_inv @ operator.Gbound

                    # Define GMRES linear operator: v -> v - h * A[j,j] * M_inv (B^T - G) v
                    n = operator.ndofs

                    # Flatten RHS for GMRES
                    rhs_flat = rhs.reshape(n)
                    if not preconditioner is None:
                        # Solve in flat space
                        Yi_flat, info = gmres(
                            gmres_operator,
                            rhs_flat,
                            M=preconditioner,
                            rtol=1e-10,
                            atol=1e-14,
                            maxiter=50,
                        )
                    else:
                        # Solve in flat space
                        Yi_flat, info = gmres(
                            gmres_operator,
                            rhs_flat,
                            rtol=1e-10,
                            atol=1e-14,
                            maxiter=50,
                        )

                    # Back to DG vector shape
                    Yi = Yi_flat.reshape(n, 1)
                    
                    K_stages[j] = operator.M_inv @ ((operator.B - operator.G) @ Yi - operator.Gbound)

            # Final combination: sum_j b[j] * K_stages[j]
            final_sum = np.zeros_like(A0)
            for j in range(self.stage):
                final_sum += self.b[j] * K_stages[j]

            A1 = A0 + h * final_sum

        else:
            raise ValueError("General implicit methods are not implemented.")
        
        return A1
    