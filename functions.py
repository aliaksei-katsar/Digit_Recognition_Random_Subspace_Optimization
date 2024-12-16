import numpy as np


#functions to distinguish between 0 and 1

def sigmoid(x) -> float:
    return 1 / (1 + np.exp(-x))


def loss_function(wb: np.array, x: np.array, y: np.array) -> float:
    w = wb[:-1]
    b = wb[-1]
    return -1 / len(x) * np.sum(y[i] * np.log(sigmoid(np.dot(x[i], w) + b))
                                + (1 - y[i]) * np.log(1 - sigmoid(np.dot(x[i], w) + b))
                                for i in range(len(x)))


def grad_loss_function(wb: np.array, x: np.array, y: np.array) -> np.array:
    w = wb[:-1]
    b = wb[-1]
    return -1 / len(x) * np.sum((y[i] - sigmoid(np.dot(w, x[i]) + b)) * np.append(x[i], 1)
                                for i in range(len(x)))


def hessian_loss_function(wb: np.array, x: np.array) -> np.array:
    w = wb[:-1]
    b = wb[-1]
    return np.reshape(np.sum(np.exp(-np.dot(w, xi) - b) / (1 + np.exp(-np.dot(w, xi) - b)) ** 2 *
                             np.outer(np.append(xi, 1), np.append(xi, 1)) for xi in x), (len(wb), len(wb)))


#functions to distinguish between all digits using softmax

def softmax_sigmoid(w: np.array, x: np.array) -> np.array:
    exp_dots = np.exp(np.dot(w.T, x))
    return exp_dots / np.sum(exp_dots, axis=0)


def softmax_loss_function(w: np.array, x: np.array, y: np.array) -> np.array:
    return -1 / len(x) * np.sum(y * np.log(softmax_sigmoid(w, x)))


def grad_softmax_loss_function(w: np.array, x: np.array, y: np.array) -> np.array:
    return np.dot(x, (softmax_sigmoid(w, x) - y).T)
