import math
import numpy as np
from numpy.polynomial.legendre import leggauss
import copy
from numpy.linalg import norm, svd, solve
from scipy.linalg import schur, lu, solve_sylvester
from scipy.optimize import fsolve
from scipy.sparse.linalg import LinearOperator

# helpers: lanczos svd
def orthogonalize(vec, against):
    """
    Orthogonalize 'vec' against the list of vectors in 'against'.
    """
    for b in against:
        vec -= np.dot(b, vec) * b
    return vec

def lanczos_svd(A, J=20, r=5, tol=1e-10):
    """
    Approximate top-r singular values/vectors using Lanczos bidiagonalization.

    Parameters:
        A  : (m x n) matrix (numpy array)
        J  : max number of Lanczos iterations
        r  : number of singular values to compute
        tol: convergence tolerance

    Returns:
        sigma  : top-r approximate singular values
        U_k    : approximate left singular vectors (m x r)
        V_k    : approximate right singular vectors (n x r)
    """
    m, n = A.shape

    # Initialization
    v0 = np.random.randn(n)
    v0 = v0 / norm(v0)
    # print(v0)
    p0 = v0.copy()
    beta0 = 1.0
    u0 = np.zeros(m)

    V = []
    U = []
    alphas = []
    betas = [beta0]

    vj = v0
    pj = p0
    betaj = beta0
    uj = u0
    for j in range(J):
        # Step 4: v_j = p_j / beta_j
        vj = pj / betaj
        V.append(copy.deepcopy(vj))

        # Step 5: r_j = A v_j - beta_j * u_j
        rj = A.matvec(vj)
        rj -= betaj * uj

        # Step 6: Orthogonalize rj against previous uj's (optional in exact arithmetic)
        rj = orthogonalize(rj, U)     # U = [u1, u2, ..., uj]

        # Step 7: alpha_j = ||rj||
        alpha_j = norm(rj)
        alphas.append(copy.deepcopy(alpha_j))

        # Step 8: u_{j+1} = rj / alpha_j
        uj = rj / alpha_j
        U.append(copy.deepcopy(uj))

        # Step 9: p_{j+1} = A^T uj - alpha_j * vj
        pj = A.rmatvec(uj)
        pj -= alpha_j * vj

        # Step 10: Orthogonalize pj (can be done explicitly if needed)
        pj = orthogonalize(pj, V)     # V = [v0, v1, ..., vj]

        # Step 11: beta_{j+1} = ||pj||
        betaj = norm(pj)
        betas.append(copy.deepcopy(betaj))
        if betaj < tol:
            break

        # Update for next iteration
        p0 = pj
        u0 = uj

    # Form U and V matrices
    U_mat = np.column_stack(U)
    V_mat = np.column_stack(V)

    # Step 16: Construct bidiagonal matrix B
    k = len(alphas)
    B = np.zeros((k, k))
    for i in range(k):
        B[i, i] = alphas[i]
        if i < k - 1:
            B[i, i + 1] = betas[i + 1]

    # Step 17: Compute SVD of B
    U_b, sigma, Vt_b = svd(B)

    # Step 18: Compute approximate singular vectors of A
    k = min(r, k)
    U_r= U_mat @ U_b[:, :k]
    V_r = V_mat @ Vt_b[:k, :].T

    return U_r, sigma[:k], V_r.T

# helpers: kronecker svd
def kronecker_svd(A, m1, m2, n1, n2, r, method="lanczos"):
    """
    Compute the rank-r Kronecker Product SVD (KSVD) of matrix A ∈ R^{m1*m2 × n1*n2}.

    Parameters:
        A   : Input LinearOperator shape (m1 * m2, n1 * n2)
        m1, m2, n1, n2 : Dimensions such that A is block-structured accordingly
        r   : Desired rank of approximation (number of Kronecker terms)

    Returns:
        A_list : List of r matrices A_j of shape (m1, n1)
        B_list : List of r matrices B_j of shape (m2, n2)
    """
    assert A.shape == (m1 * m2, n1 * n2), "Matrix shape must match block structure"

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

    # Step 2: SVD
    if method == "lanczos":
        U, S, Vt = lanczos_svd(A_tilde, J=10, r=r, tol=1e-10)
    else:
        raise ValueError("Invalid method. Must be 'svd' or 'lanczos'.")

    A_list = []
    B_list = []

    rmin = min(r, S.shape[0])
    for k in range(rmin):
        uj = U[:, k] * np.sqrt(S[k])
        vj = Vt[k, :] * np.sqrt(S[k])

        A_j = uj.reshape((m1, n1), order='F')
        B_j = vj.reshape((m2, n2), order='F')

        A_list.append(A_j)
        B_list.append(B_j)

    return A_list, B_list, S[:rmin]

def form_2d_preconditioner(A, m1, m2, n1, n2, r):
    A_list, B_list, _ = kronecker_svd(A, m1, m2, n1, n2, r=2)
    A1, A2 = A_list
    B1, B2 = B_list

    L_A2, U_A2 = lu(A2, permute_l=True)
    L_B1, U_B1 = lu(B1, permute_l=True)

    A2_inv_A1 = solve(A2, A1)
    B1_inv_B2 = solve(B1, B2)

    T1, Q1 = schur(A2_inv_A1)
    T2, Q2 = schur(B1_inv_B2)

    return {
        "A1": A1, "B1": B1, "A2": A2, "B2": B2,
        "Schur_A": (Q1, T1),
        "LU_A2": (L_A2, U_A2),
        "LU_B1": (L_B1, U_B1),
        "Schur_B": (Q2, T2)
    }
    
def apply_2d_preconditioner(b, preconditioner, m2=None, n2=None):
    """
    Solve (A1 ⊗ B1 + A2 ⊗ B2) x = b using the 2D Kronecker preconditioner
    via Schur-based Sylvester solve.

    Parameters
    ----------
    b : (N,) array
        Right-hand side vector, N = dim(A2)*dim(B1).
    preconditioner : dict
        Output of form_2d_preconditioner.
    m2, n2 : unused
        Kept for API compatibility with your existing test.
    """
    # Unpack
    L_A2, U_A2 = preconditioner["LU_A2"]
    L_B1, U_B1 = preconditioner["LU_B1"]
    Q1, T1 = preconditioner["Schur_A"]
    Q2, T2 = preconditioner["Schur_B"]

    # Dimensions
    p = U_A2.shape[0]   # dim(A2)
    q = U_B1.shape[0]   # dim(B1)
    assert b.size == p * q

    # --- Step 1: RHS_mat = B1^{-1} * B_mat * A2^{-T} ---
    # Interpret b as vec_F(B_mat) with shape (q x p)
    B_mat = b.reshape(q, p, order="F")

    # Solve B1 * X = B_mat  →  X = B1^{-1} B_mat
    tmp = solve(L_B1, B_mat)        # forward
    tmp = solve(U_B1, tmp)          # backward

    # We want RHS_mat = tmp * A2^{-T}.
    # This is equivalent to solving A2 * Y^T = tmp^T, then Y = RHS_mat.
    tmp_T = tmp.T
    Y_T = solve(L_A2, tmp_T)        # forward with A2
    Y_T = solve(U_A2, Y_T)          # backward with A2
    RHS_mat = Y_T.T                 # = tmp * A2^{-T}

    # --- Step 2: Transform to Schur basis ---
    # C2 X + X C1^T = RHS_mat
    # with C1 = A2^{-1} A1 = Q1 T1 Q1^T, C2 = B1^{-1} B2 = Q2 T2 Q2^T
    #
    # Let X = Q2 Y Q1^T. Then
    #   T2 Y + Y T1^T = Q2^T RHS_mat Q1
    RHS_hat = Q2.T @ RHS_mat @ Q1

    # --- Step 3: Solve Sylvester T2 Y + Y T1^T = RHS_hat ---
    Y = solve_sylvester(T2, T1.T, RHS_hat)

    # --- Step 4: Back-transform X = Q2 Y Q1^T ---
    X = Q2 @ Y @ Q1.T

    # Return vec_F(X)
    x = X.reshape(p * q, order="F")
    return x