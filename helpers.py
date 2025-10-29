import numpy as np

def print_matrix(name, mat):
    print(f"\n{name}: shape={mat.shape}")
    with np.printoptions(precision=4, suppress=True):
        print(mat)