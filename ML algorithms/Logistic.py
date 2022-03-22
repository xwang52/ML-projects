import numpy as np
import numpy.linalg as la

"""
The following is a simple implementation of Logistic Regression using gradient descent algorithm. The logistic
regression has two sets of parameters, W the coefficients and b the intercept. The gradient of each parameter is 
calculated to update the parameter in the opposite direction of the gradient. A step size alpha is used to control
the rate at which the update is carried out. The process is continued iteratively until the parameters do not get updated
(or equivalently the gradient vanishes) or the number of iteration is reached. 

The implementation aims at replicating the algorithm used in scikit-learn, and no opitmization is used, so the algorithm will be
slower. 
"""

logit = lambda x: 1/(1 + np.exp(-x))

class Logistic():
  def __init__(self, alpha = 0.01, n_iter = 500, seed = 123, tol = 1e-4):
    self.alpha = alpha
    self.n_iter = n_iter
    self.seed = seed
    self.tol = tol
    self.coeff_ = None

  def fit(self, X, y):
    m, n = X.shape
    y = y.reshape(y.shape[0], 1)
    np.random.seed(self.seed)
    W = np.random.rand(n,1)
    b = np.random.rand(1)[0]
    dW = np.dot(X.T, y - logit(np.dot(X, W)))*(-1/m)
    db = np.sum(y - logit(np.dot(X, W)))*(-1/m)
    for i in range(self.n_iter):
      W -= self.alpha * dW
      b -= self.alpha * db
      dW = np.dot(X.T, y - logit(np.dot(X, W)))*(-1/m)
      db = np.sum(y - logit(np.dot(X, W)))*(-1/m)
      if la.norm(dW) + la.norm(db) < self.tol:
        break
    self.coeff_ = (W, b) 
  
  def predict(self, X):
    return (logit(np.dot(X, self.coeff_[0]) + self.coeff_[1]) > 0.5).reshape(-1)*1
  
  def predict_proba(self, X):
    return (logit(np.dot(X, self.coeff_[0]) + self.coeff_[1])).reshape(-1)
