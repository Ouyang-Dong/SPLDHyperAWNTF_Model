import math
import numpy as np
import hypergraph_utils

def constructHW(X):

    H_w = hypergraph_utils.construct_H_with_KNN(X)

    """incidence matrix"""
    H = np.ones([H_w.shape[0],H_w.shape[1]])
    row_num = H.shape[0]
    col_num = H.shape[1]
    for i in range(row_num):
        for j in range(col_num):
            if H_w[i,j] != 0:
                H[i,j] = 1
            else:
                H[i,j] = 0

    """affinity matrix"""
    row_dim = X.shape[0]
    tol_sum = 0
    for t in range(row_dim):
        tol_sum = tol_sum + X[t,:]
    avg_mi = tol_sum / row_dim
    temp_list = []
    for i in range(row_dim):
        temp_list.append(math.pow(np.linalg.norm((X[i,:] - avg_mi),ord = 2),2))
    sigma = np.sqrt(sum(temp_list) / (row_dim - 1))

    A = np.ones([row_dim,row_dim])
    for m in range(row_dim):
        for n in range(row_dim):
            dist = math.pow(np.linalg.norm((X[m,:] - X[n,:]),ord = 2),2)
            A[m,n] = np.exp(-dist / (sigma**2))


    row_num = A.shape[0]
    col_num = A.shape[1]

    """the weight of each hyperedge"""
    W = np.zeros([1,col_num])
    w = 0
    for c in range(col_num):
        for r in range(row_num):
            if H[r,c] != 0:
                w = w + A[r,c]
        W[:,c] = w
        w = 0

    DV,S = hypergraph_utils._generate_G_from_H(H, W[0])

    return DV,S