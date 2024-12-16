import numpy as np
from armijo import armijo
import functools


#global newton method
def global_newton(f: functools.partial, gradf: functools.partial, hessianf: functools.partial,
                  x, eps=1e-3, iter=1000, gamma=0.01, beta=0.5):

    #initializing constant values
    alpha1 = 1e-8
    alpha2 = 1e-4
    p = 0.2

    #finding function values for given x
    current_grad = gradf(x)
    current_hess = hessianf(x)
    current_grad_norm = np.linalg.norm(current_grad)

    current_iter = 1
    while current_grad_norm > eps and current_iter <= iter:

        #try to find the right direction by solving system of linear equations
        try:
            d = np.linalg.solve(current_hess, -1 * current_grad)
            #checking if found direction better than -gradient
            if (-np.dot(current_grad, d) > np.min([alpha1, alpha2 * (np.power(current_grad_norm, p))]) *
                    current_grad_norm * np.linalg.norm(d)):
                s = d
            else:
                s = -current_grad
        except np.linalg.LinAlgError:
            s = -current_grad

        #finding learning rate with armijo rule and updating all functions values at new point
        sigma = armijo(f, x, current_grad, s, gamma, beta)
        x = x + sigma * s
        current_grad = gradf(x)
        current_hess = hessianf(x)
        current_grad_norm = np.linalg.norm(current_grad)
        current_iter += 1

    return x
