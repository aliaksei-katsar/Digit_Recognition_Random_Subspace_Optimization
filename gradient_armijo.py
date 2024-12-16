
import functools
import numpy as np
from armijo import armijo
from power_wilfe import power_wilfe


#standard gradient_descent using armijo/power wilfe/constant learning rate
def gradient_descent(f: functools.partial, gradf: functools.partial, xk: np.array, eps=1e-6, iter=1000) -> np.array:

    current_iter = 0
    while np.linalg.norm(gradf(xk)) > eps and current_iter <= iter:

        current_iter += 1
        gradf_x = gradf(xk)
        #sigma = power_wilfe(f, gradf, xk, gradf_x, -gradf_x, 0.0001)
        #sigma = armijo(f, xk, gradf_x, -gradf_x, 0.01, 0.5)
        sigma = 0.2
        xk -= sigma * gradf_x
        print("iteration: ", current_iter)
        print("Current gradient norm value: ", np.linalg.norm(gradf(xk)))
        print("Current function value is: ", f(xk))

    return xk
