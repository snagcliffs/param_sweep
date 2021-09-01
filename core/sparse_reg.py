import numpy as np
import numba as nb
import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPerformanceWarning, NumbaWarning

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

@nb.jit(parallel=True)
def FoBaGreedy(A, y, tau, args = [100,5,1,True]):
    """
    Forward-Backward greedy algorithm for sparse regression.

    For more details see:
    Zhang, Tong. 'Adaptive Forward-Backward Greedy Algorithm for Sparse Learning with Linear Models', NIPS, 2008
    
    For relearn option, see github code for:
    Thaler et al. 'Sparse identification of truncation errors,' JCP, 2019

    Inputs:
        A,y   : from linear system Ax=y
        tau   : sparsity parameter
        args  : method specific arguments including:
          maxit_f : max forward iterations
          maxit_b : max backward iterations per backward call
          backwards_freq : frequency of backwards method calls
          relearn : see lines 48-59
    Returns:
        x     : Sparse approximation to A^{-1}y
        delta_tau : minimal change in tau to affect result
    """

    maxit_f, maxit_b, backwards_freq, relearn = args

    n,d = A.shape
    F = {}
    F[0] = set()
    x = {}
    x[0] = np.zeros((d,1))
    k = 0
    delta = {}
    
    # We initially assume delta_tau is infinite and lower as needed
    delta_tau = np.inf
    
    for forward_iter in range(maxit_f):

        k = k+1

        # forward step
        zero_coeffs = np.where(x[k-1] == 0)[0]
        err_after_addition = []
        residual = y - A.dot(x[k-1])
        for i in zero_coeffs:

            if relearn:
                F_trial = F[k-1].union({i})
                x_added = np.zeros((d,1))
                x_added[list(F_trial)] = np.linalg.lstsq(A[:, list(F_trial)], y, rcond=None)[0]
                
            else:
                # Per figure 3 line 8 in Zhang, do not retrain old variables.
                # Only look for optimal alpha, which is solving for new x if 
                # and only if columns of $A$ are orthogonal
                alpha = A[:,i].T.dot(residual)/np.linalg.norm(A[:,i])**2
                x_added = np.copy(x[k-1])
                x_added[i] = alpha
            
            err_after_addition.append(np.linalg.norm(A.dot(x_added)-y))
            
        i = zero_coeffs[np.argmin(err_after_addition)]
        
        F[k] = F[k-1].union({i})
        x[k] = np.zeros((d,1))
        x[k][list(F[k])] = np.linalg.lstsq(A[:, list(F[k])], y, rcond=None)[0]

        # If improvement is sufficiently small, return last estimate
        delta[k] = np.linalg.norm(A.dot(x[k-1]) - y)**2 - np.linalg.norm(A.dot(x[k]) - y)**2
        if delta[k] <= tau:            
            return x[k-1], delta_tau
        
        # Otherwise, how much larger would tolerance need to be to stop?
        delta_tau = np.min([delta_tau, delta[k]-tau])

        # backward step, do once every backwards_freq forward stdelta_tau
        if forward_iter % backwards_freq == 0 and forward_iter > 0:
            
            dk = delta[k]

            for backward_iter in range(maxit_b):

                non_zeros = np.where(x[k] != 0)[0]
                err_after_simplification = []
                for j in non_zeros:
                    
                    if relearn:
                        F_trial = F[k].difference({j})
                        x_simple = np.zeros((d,1))
                        x_simple[list(F_trial)] = np.linalg.lstsq(A[:, list(F_trial)], y, rcond=None)[0]
                
                    else:
                        x_simple = np.copy(x[k])
                        x_simple[j] = 0
                        
                    err_after_simplification.append(np.linalg.norm(A.dot(x_simple) - y)**2)
                    
                j = np.argmin(err_after_simplification)

                # check for break condition on backward step
                # how much does error increase when subtracting a term?
                delta_p = err_after_simplification[j] - np.linalg.norm(A.dot(x[k]) - y)**2
                
                # Original cutoff from paper is based on improvement of kth term
                if delta_p > 0.5*delta[k]: break

                # Optionally, we can use the improvement from the last term added
                # if delta_p > 0.5*dk: break

                k = k-1;
                F[k] = F[k+1].difference({j})
                x[k] = np.zeros((d,1))
                x[k][list(F[k])] = np.linalg.lstsq(A[:, list(F[k])], y, rcond=None)[0]
                
                if k == 0: break

        if np.count_nonzero(x[k]) ==  x[k].size: break

    return x[k], delta_tau
    
@nb.jit(nopython=True)
def STRidge(A, y, tau, args=1e-5):
    """
    Sequential Threshold Ridge Regression algorithm for finding sparse approximation to x = A^{-1}y.
    If not none, C is symmetric positive definite weight matrix.

    Inputs:
        A,y   : from linear system Ax=y
        tau   : sparsity parameter
        args  : method specific arguments including:
          lam : ridge penalty
    Returns:
        x     : Sparse approximation to A^{-1}y
        delta_tau : minimal change in tau to affect result
    """
    
    m,n = A.shape
    lam = args

    # Solve least squares problem
    x = np.linalg.solve(A.T @ A + lam*np.eye(n), A.T @ y) 

    # Threshold
    G = np.where(np.abs(x) > tau)[0]    # Set of active terms
    Gc = np.where(np.abs(x) <= tau)[0]  # Complimentary set (to be removed)
    
    if len(G) != 0: delta_tau = np.min(np.abs(x[G])) - tau
    else: delta_tau = np.inf
    
    if len(Gc) == 0: 
        # No terms have been removed
        return x, delta_tau

    else:
        # Terms were removed
        xG, delta_tau_tilde = STRidge(A[:,G], y, tau, args)    
    
        for j in range(len(G)): x[G[j]] = xG[j]
        for j in Gc: x[j] = 0
        delta_tau = np.min(np.array([delta_tau, delta_tau_tilde]))
        
        return x, delta_tau


