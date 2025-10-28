import math
import numpy as np
from numpy.polynomial.legendre import leggauss
import copy
from numpy.linalg import norm, svd, solve
from scipy.optimize import fsolve


# helpers: lanczos svd
def orthogonalize(vec, A, against):
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
    v0 = np.ones(n)
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
        A.TildeVmult(rj ,vj)
        rj -= betaj * uj

        # Step 6: Orthogonalize rj against previous uj's (optional in exact arithmetic)
        rj = orthogonalize(rj, A, U)     # U = [u1, u2, ..., uj]

        # Step 7: alpha_j = ||rj||
        alpha_j = norm(rj)
        alphas.append(copy.deepcopy(alpha_j))

        # Step 8: u_{j+1} = rj / alpha_j
        uj = rj / alpha_j
        U.append(copy.deepcopy(uj))

        # Step 9: p_{j+1} = A^T uj - alpha_j * vj
        A.TildeTvmult(pj, uj)
        pj -= alpha_j * vj

        # Step 10: Orthogonalize pj (can be done explicitly if needed)
        pj = orthogonalize(pj, A, V)     # V = [v0, v1, ..., vj]

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
    U_r= U_mat @ U_b[:, :r]
    V_r = V_mat @ Vt_b[:r, :].T

    return U_r, sigma[:r], V_r.T

# helpers: kronecker svd
def kronecker_svd(A, m1, m2, n1, n2, r, method="svd"):
    """
    Compute the rank-r Kronecker Product SVD (KSVD) of matrix A ∈ R^{m1*m2 × n1*n2}.

    Parameters:
        A   : Input matrix of shape (m1 * m2, n1 * n2)
        m1, m2, n1, n2 : Dimensions such that A is block-structured accordingly
        r   : Desired rank of approximation (number of Kronecker terms)

    Returns:
        A_list : List of r matrices A_j of shape (m1, n1)
        B_list : List of r matrices B_j of shape (m2, n2)
    """
    assert A.shape == (m1 * m2, n1 * n2), "Matrix shape must match block structure"

    # Step 2: SVD
    if method == "svd":
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
    elif method == "lanczos":
        U, S, Vt = lanczos_svd(A, J=20, r=r, tol=1e-10)
    else:
        raise ValueError("Invalid method. Must be 'svd' or 'lanczos'.")

    A_list = []
    B_list = []

    for k in range(r):
        uj = U[:, k] * np.sqrt(S[k])
        vj = Vt[k, :] * np.sqrt(S[k])

        A_j = uj.reshape((m1, n1), order='F')
        B_j = vj.reshape((m2, n2), order='F')

        A_list.append(A_j)
        B_list.append(B_j)

    return A_list, B_list, S[:r]