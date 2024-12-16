import numpy as np
import function
import armijo


#checking armijo condition for given values
def armijo_condition(f: function, x: np.array, fx: np.array, s: np.array, sigma: float, slope: float):
    if f(x + sigma * s) - fx <= sigma * slope:
        return True
    return False


#checking power wilfe condition for given values
def power_wilfe_condition(gradf: function, x: np.array, fx: np.array, s: np.array, sigma: float, slope: float):
    if np.dot(gradf(x + sigma * s).flatten(), s.flatten()) - fx >= slope:
        return True
    return False


#finding learning rate
def power_wilfe(f: function, gradf: function, x: np.array, gradf_x: np.array, s: np.array, gamma=0.01, nu=0.9,
                beta=0.5) -> float:
    sigma = 1
    fx = f(x)
    slope = np.dot(gradf_x.flatten(), s.flatten())
    slope_gamma = gamma * slope
    slope_nu = nu * slope

    #standard power wilfe algorithm
    if armijo_condition(f, x, fx, s, sigma, slope_gamma):
        if power_wilfe_condition(gradf, x, fx, s, sigma, slope_nu):
            return sigma
        else:
            while armijo_condition(f, x, fx, s, sigma, slope_gamma):
                sigma /= beta
            sigma_plus = sigma
            sigma_minus = sigma * beta
    else:
        sigma_minus = armijo.armijo(f, x, gradf_x, s, gamma, beta)
        sigma_plus = sigma_minus / beta
        sigma = sigma_minus

    iter = 0
    while (not power_wilfe_condition(gradf, x, fx, s, sigma_minus, slope_nu)) and iter < 15:
        sigma = (sigma_plus + sigma_minus) / 2
        if armijo_condition(f, x, fx, s, sigma, slope_gamma):
            sigma_minus = sigma
        else:
            sigma_plus = sigma
        iter += 1

    return sigma
