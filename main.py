import numpy as np
import functions as f
import gradient_armijo as gar
import data_loading
from functools import partial
import global_newton as gn




#Distinguish between 0 and 1

#loading data
x_train, y_train, x_test, y_test = data_loading.load_data_logistic_regression()

#initializing start value as well as loss function and its gradient and hessian matrix
x_start = np.zeros(len(x_train[0]) + 1)
loss = partial(f.loss_function, x=x_train, y=y_train)
grad_loss = partial(f.grad_loss_function, x=x_train, y=y_train)
hessian_loss = partial(f.hessian_loss_function, x=x_train)

#running gradient descent using Armijo/Powell Wolfe/constant learning rate
sol = gar.gradient_descent(loss, grad_loss, x_start, 1e-1)

#trying global newton method, which doesn't work because hessian matrix is always singular
#sol = gn.global_newton(loss, grad_loss, hessian_loss, x_start)

#trying solution on test set
correct = 0
for (xi, yi) in zip(x_test, y_test):
    v = np.dot(xi, sol[:-1]) + sol[-1]
    if v >= 0:
        if yi == 1:
            correct += 1
    else:
        if yi == 0:
            correct += 1

print("right decisions: ", correct)
print("all decisions: ", len(x_test))
print("Accuracy score: ", correct / len(y_test))


#distinguish between all digits using softmax

#loading data
x_train, y_train, x_test, y_test = data_loading.load_data_softmax()

#initializing start value as well as loss function and its gradient
x_start = np.zeros((x_train.shape[0], y_train.shape[0]))
loss = partial(f.softmax_loss_function, x=x_train, y=y_train)
grad_loss = partial(f.grad_softmax_loss_function, x=x_train, y=y_train)

#running gradient descent using Armijo/Powell Wolfe/constant learning rate
sol = gar.gradient_descent(loss, grad_loss, x_start, 1e-1, 1e2)

#trying solution on test set
correct = 0
for (xi, yi) in zip(x_test.T, y_test):
    v = f.softmax_sigmoid(sol, xi)
    pred = np.argmax(v)
    if yi == pred:
        correct += 1

print("right decisions: ", correct)
print("all decisions: ", len(y_test))
print("Accuracy score: ", correct / len(y_test))
