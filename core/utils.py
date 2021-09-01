import numpy as np
from scipy.integrate import odeint
import scipy.sparse as sparse
np.random.seed(0)

def FiniteDifferences(X, dt, order, axis = 0):
    """
    order should be in [2,4,6]
    """
    
    if axis == 1: return FiniteDifferences(X.T, dt, order).T
    
    dX = np.zeros_like(X)
    m,n = X.shape
    
    w = int(order/2)
    
    central_templates = {2 : np.array([-1/2,0,1/2]),\
                         4 : np.array([1/12,-2/3,0,2/3,-1/12]),\
                         6 : np.array([-1/60,3/20,-3/4,0,3/4,-3/20,1/60])}
    
    forward_templates = {2 : np.array([-3/2,2,-1/2]),\
                         4 : np.array([-25/12,4,-3,4/3,-1/4]),\
                         6 : np.array([-49/20,6,-15/2,20/3,-15/4,6/5,-1/6])}
    
    for i in range(n):
        for j in range(w, m-w):
            dX[j,i] = np.inner(X[j-w:j+w+1,i], central_templates[order])
            
        for j in range(w):
            dX[j,i] = np.inner(X[j:j+2*w+1,i], forward_templates[order])
            dX[m-j-1,i] = np.inner(X[m-j-2*w-1:m-j,i], -forward_templates[order][::-1])
            
    return dX/dt

def TikhonovDiff(f, dx, lam):
    """
    Tikhonov differentiation.

    return argmin_g ||Ag-f||_2^2 + lam*||Dg||_2^2
    where A is trapezoidal integration and D is finite differences for first dervative
    """

    m = f.shape[0]
    f = np.matrix(f - f[0,...])

    # Trapezoidal approximation to an integral
    A = np.zeros((m,m))
    for i in range(1, m):
        A[i,i] = dx/2
        A[i,0] = dx/2
        for j in range(1,i): A[i,j] = dx
    
    e = np.ones(m-2)
    D = sparse.diags([e, -2*e, e], [0, 1, 2], shape=(m-2, m)).todense() / dx**2
    
    # Invert to find derivative
    g = np.array(np.linalg.lstsq(A.T.dot(A) + lam*D.T.dot(D),A.T.dot(f),rcond=None)[0])
    
    return g

def deg_p_polynomials(n,deg):
    """
    Given n and deg, return all coefficient lists of monomials in n variables of degree deg.
    Example: deg_p_polynomials(2,3) returns [[3,0],[2,1],[1,2],[0,3]]
    """
    
    if deg == 0: return [[0 for _ in range(n)]]
    if n == 1: return [[deg]]

    polys = []    
    for j in range(deg+1):
        smaller = deg_p_polynomials(n-1,deg-j)
        for small_poly in smaller:
            polys.append([j]+small_poly)
    return polys

def polynomials(n,max_deg):
    """
    Given n and max_deg, return all coefficient lists of monomials in n variables of degree <= max_deg.
    Example: deg_p_polynomials(2,3) returns [[0,0],[1,0],[0,1],[2,0],[1,1],[0,2],[3,0],[2,1],[1,2],[0,3]]
    """
    
    polys = []
    for p in range(0,max_deg+1):
        polys = polys + deg_p_polynomials(n,p)
        
    return polys

def polynomial_feature_maps(n, max_deg):
    
    powers = polynomials(n,max_deg)
    def f(x,y): return np.prod(np.power(x, y), axis = 1)
    
    polynomial_features = [lambda x, y = np.array(power): f(x,y).reshape(x.shape[0],1) for power in powers]
    
    descriptions = [np.array(power) for power in powers]
        
    return polynomial_features, descriptions

def lorenz_ode(X, t, params):
    s,r,b = params
    x,y,z = X
    return [s*(y-x), x*(r-z)-y, x*y-b*z]

def get_lorenz_data(m = 1001, T=10, noise_percent=1, diff = 'Tikhonov', p=5):

    n = 3
    t = np.linspace(0,T,m); dt = (t[1]-t[0])

    # Lorenz 63 data
    params = (10,28,8.0/3)
    X = odeint(lorenz_ode, [5,5,15], t, args=(params,), rtol = 1e-12, atol = 1e-12)
    sigma = [0.01*noise_percent*np.std(X[:,j]) for j in range(n)]
    X = X + np.hstack([sigma[j]*np.random.randn(m,1) for j in range(n)])

    if type(diff) == int: y = FiniteDifferences(X, dt, diff)
    else: y = TikhonovDiff(X, dt, 1e-10)

    # Features
    feature_maps, feature_descriptions = polynomial_feature_maps(n, p)
    Theta = np.hstack([f(X) for f in feature_maps])

    # True model
    x_true = np.zeros((len(feature_maps),n))
    x_true[2,0] = 10
    x_true[3,0] = -10
    x_true[3,1] = 28
    x_true[7,1] = -1
    x_true[2,1] = -1
    x_true[8,2] = 1
    x_true[1,2] = -8/3

    return X, y, Theta, x_true

def shift(x, s):
    if s == 0: return x
    else: return np.concatenate([x[s:], x[:s]])
    
def L96(x,t,params):
    F = params
    return (shift(x,1)-shift(x,-2))*shift(x,-1)-x+F

def get_l96_data(m = 201, T=10, noise_percent=1, diff = 'Tikhonov', p=2, n=20, F=16, dim=0):

    # Lorenz 96 data
    t = np.linspace(0,T,m); dt = (t[1]-t[0])
    x0 = np.exp(-(np.arange(n)-n/2)**2 / 16)
    X = odeint(L96, x0, t, (F,))
    sigma = [0.01*noise_percent*np.std(X[:,j]) for j in range(n)]
    X = X + np.hstack([sigma[j]*np.random.randn(m,1) for j in range(n)])

    if type(diff) == int: y = FiniteDifferences(X[:,dim].reshape(m,1), dt, diff)
    else: y = TikhonovDiff(X[:,dim].reshape(m,1), dt, 1e-10)

    # Features
    feature_maps, feature_descriptions = polynomial_feature_maps(n, p)
    Theta = np.hstack([f(X) for f in feature_maps])
    d = len(feature_maps)

    # True predictor
    e_dim = np.zeros(n); e_dim[dim]=1  # linear term in this dimension
    e_dim_pm1 = np.zeros(n); e_dim_pm1[(dim+1)%n]=1; e_dim_pm1[(dim-1)%n]=1  # quadratic x_{dim +- 1}
    e_dim_m2m1 = np.zeros(n); e_dim_m2m1[(dim-2)%n]=1; e_dim_m2m1[(dim-1)%n]=1  # quadratic x_{dim - 2}x_{dim - 1}
    c_dim = np.where(np.array([np.all(feature_descriptions[j] == \
                                      e_dim.astype(int)) for j in range(d)]).astype(int) == 1)[0]
    c_dim_pm1 = np.where(np.array([np.all(feature_descriptions[j] == \
                                      e_dim_pm1.astype(int)) for j in range(d)]).astype(int) == 1)[0]
    c_dim_m2m1 = np.where(np.array([np.all(feature_descriptions[j] == \
                                      e_dim_m2m1.astype(int)) for j in range(d)]).astype(int) == 1)[0]
    x_true = np.zeros((len(feature_maps),1))
    x_true[0,0] = F
    x_true[c_dim,0] = -1
    x_true[c_dim_pm1,0] = 1 
    x_true[c_dim_m2m1,0] = -1
    
    return X, y, Theta, x_true

def get_random_data(m=100, n=100, s=10, noise_percent=10, A_cond=10, p=1):

    # Random matrix A with cond(A) = A_cond
    U,_,Vt = np.linalg.svd(np.random.randn(m,n), full_matrices=False)
    S = np.exp(np.linspace(0,np.log(A_cond),np.min([n,m])))
    A = U @ np.diag(S) @ Vt

    # Get polynomial features
    if p != 1:
        feature_maps, feature_descriptions = polynomial_feature_maps(n, p)
        A = np.hstack([f(A) for f in feature_maps])
        n = A.shape[1]

    # Random 'true' weights for sparse linear model
    nnz = np.random.choice(n,s)
    x_true = np.zeros((n,1))
    x_true[nnz] = np.random.randn(s,1)

    y_true = A.dot(x_true)
    y = y_true + 0.01*noise_percent*np.std(y_true)*np.random.randn(m,1)

    return A,y,x_true