import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
advertising_data = pd.read_csv("/Users/artemkarpuhov/iu1-2-course-2023/iu1-2-course-2023/дз/дз1/advertising.csv")

#task 1:
#print(advertising_data.head())
X = np.array(advertising_data.values[:, :3])
Y = np.array(advertising_data.values[:, 3])
x_scalled_means = np.mean(X, axis=0)
x_scalled_std = np.std(X, axis=0)
X = ((X-x_scalled_means)/x_scalled_std)
M = np.ones(X.shape[0])
Column_1 = M[:, None]
X = np.hstack((X, Column_1))

#task 2:
def mseerror(y, y_pred):
    mse = np.square(np.subtract(y, y_pred)).mean()
    return mse
answer1 = mseerror(Y, np.median(Y))
print(answer1)

# task 3:
def normal_equation(X, y):
    w = np.dot(np.dot(X.T, y), np.linalg.inv(np.dot(X.T, X)))
    return w
norm_eq_weights = normal_equation(X, Y)
#print(norm_eq_weights)
answer2 = np.dot(norm_eq_weights, np.mean(X, axis=0))
#print(round(answer2, 3))

# task 4:
def linear_prediction(X, W):
    pred = np.dot(X,W)
    return(pred)
answer3 = mseerror(Y, linear_prediction(X, norm_eq_weights))
#print(round(answer3, 3))
def stochastic_gradient_step(X, y, w, train_ind, eta=0.01):
    grad0 = 2 * X[train_ind][0] * (sum(w * X[train_ind])-y[train_ind]) / X.shape[0]
    grad1 = 2 * X[train_ind][1] * (sum(w * X[train_ind])-y[train_ind]) / X.shape[0]
    grad2 = 2 * X[train_ind][2] * (sum(w * X[train_ind])-y[train_ind]) / X.shape[0]
    grad3 = 2 * X[train_ind][3] * (sum(w * X[train_ind])-y[train_ind]) / X.shape[0]
    return  w - eta * np.array([grad0, grad1, grad2, grad3])

# task 5:
def stochastic_gradient_descent(X, y, w_init, eta=1e-2, max_iter=1e5):
  w = w_init
  errors = []
  iter_num = 0
  while iter_num < max_iter:
    new_ind = np.random.randint(X.shape[0])
    w_new = stochastic_gradient_step(X, Y, w, new_ind, eta)
    w = w_new
    errors.append(mseerror(Y, linear_prediction(X, w)))
    iter_num+=1
  return w, errors

stoch_grad_desc_weights, stoch_errors_by_iter = stochastic_gradient_descent(X, Y, np.zeros(X.shape[1]), eta=0.01)

plt.plot(range(50), stoch_errors_by_iter[:50])
plt.xlabel('Iteration number')
plt.ylabel('MSE')
#plt.show()

plt.plot(range(len(stoch_errors_by_iter)), stoch_errors_by_iter)
plt.xlabel('Iteration number')
plt.ylabel('MSE')
plt.show()

#print(stoch_grad_desc_weights)

#print(stoch_errors_by_iter[-1])

answer4 = mseerror(Y, np.dot(X, stoch_grad_desc_weights))
#print(answer4)