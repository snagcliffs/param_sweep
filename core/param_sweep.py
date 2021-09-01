import numpy as np
from sparse_reg import *

def Search_tau(A, y, S, args, normalize=True, min_delta=0):
    """
    Complete parameter search for sparse regression method S.

    Input:
        A,y   : from linear system Ax=y
        S     : sparse regression method
        args  : arguments for sparse regression method
        normalize : boolean.  Normalize columns of A?
        min_delta : minimum change in tau

    Returns:
        X     : list of all possible outputs of S(A,y,tau)
        Tau   : list of values of tau corresponding to each x in X
    """

    X = []
    Tau =[]
    tau = 0
    
    # Normalize
    if normalize:
        normA = np.linalg.norm(A,axis=0)
        A = A @ np.diag(normA**-1)

    for j in range(2**A.shape[1]):
        
        # Apply sparse regression
        x, delta_tau = S(A, y, tau, args)
        delta_tau = np.max([delta_tau, min_delta])
        X.append(x)
        Tau.append(tau)

        # Break condition
        if np.max(np.abs(x)) == 0 or delta_tau == np.inf: break

        # Update tau
        tau = tau+delta_tau
            
    # Renormalize x
    if normalize:
        X = [np.diag(normA**-1) @ x for x in X]
    
    return X,Tau