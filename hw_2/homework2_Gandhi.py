from doctest import testfile
from unittest import result
from matplotlib import pyplot as plt, widgets

import numpy as np

########################################################################################################################
# PROBLEM 2
########################################################################################################################
# Given a vector x of (scalar) inputs and associated vector y of the target labels, and given
# degree d of the polynomial, train a polynomial regression model and return the optimal weight vector.
def trainPolynomialRegressor (x, y, d):
    
    X = np.ones((d+1,x.shape[0]))
    
    col = 0
    for el in x:
        x_ = np.array(())
        for i in range(d+1):
            x_ = np.append(x_, np.array(([el**i])), axis=0)
        X[:,col] = x_ 
        col += 1
    
    weights = method1(X, y)
    return(weights)

########################################################################################################################
# PROBLEM 1
########################################################################################################################

# Given an array of faces (N x M x M, where N is number of examples and M is number of pixes along each axis),
# return a design matrix Xtilde ((M**2 + 1) x N) whose last row contains all 1s.
def reshapeAndAppend1s (faces):
    N = faces.shape[0]
    M = faces.shape[1]**2
    Xtilde = faces.reshape(N, M)
    Xtilde = np.transpose(Xtilde)
    ones = np.ones((1, N))
    Xtilde = np.append(Xtilde, ones, axis=0)
    return Xtilde

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, return the (unregularized)
# MSE.
def fMSE (wtilde, Xtilde, y):
    inner_math = Xtilde.T.dot(wtilde) - y
    inner_math = inner_math**2
    cost = (np.mean(inner_math)/2)
    return cost

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, and a regularization strength
# alpha (default value of 0), return the gradient of the (regularized) MSE loss.
def gradfMSE (wtilde, Xtilde, y, alpha = 0.):
    
    grad_fMSE_inner_math = Xtilde.T.dot(wtilde) - y
    grad_fMSE_inner_math = Xtilde.dot(grad_fMSE_inner_math)
    penalty = (alpha*wtilde)/Xtilde.shape[1]

    grad_fMSE = grad_fMSE_inner_math + penalty

    grad_fMSE = (grad_fMSE)/Xtilde.shape[1]
    
    return grad_fMSE

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using the analytical solution.
def method1 (Xtilde, y):
    weights = np.linalg.solve(Xtilde.dot(Xtilde.T), Xtilde.dot(y))
    return weights

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE.
def method2 (Xtilde, y):
    weigths = gradientDescent(Xtilde, y)
    return weigths

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE
# with regularization.
def method3 (Xtilde, y):
    ALPHA = 0.1
    weigths = gradientDescent(Xtilde, y, alpha=ALPHA)
    return weigths

# Helper method for method2 and method3.
def gradientDescent (Xtilde, y, alpha = 0.):
    EPSILON = 3e-3  # Step size aka learning rate
    T = 5000  # Number of gradient descent iterations

    w = 0.01*np.random.rand(2305)
    for i in range(T):
        
        grad = gradfMSE(w, Xtilde, y, alpha)

        w = w - EPSILON*grad

    return w

if __name__ == "__main__":
    # Load data
    Xtilde_tr = reshapeAndAppend1s(np.load("age_regression_Xtr.npy"))
    ytr = np.load("age_regression_ytr.npy")
    Xtilde_te = reshapeAndAppend1s(np.load("age_regression_Xte.npy"))
    yte = np.load("age_regression_yte.npy")

    unity_Xx = np.array(([-1.89, 1.19, -15.3, -18.1],[1, 1, 1, 1]))
    unity_Xz = np.array(([-1.65, -11.45, -15.7, -6.2],[1, 1, 1, 1]))
    # unity_Y = np.array(([14.23, 13.85, -4.33, -4.29], [9.22, -9.15, -9.15, 8.45]))
    unity_Yx = np.array(([14.23, 13.85, -4.33, -4.29]))
    # print(unity_X)
    # print(unity_Y)

    w1 = method1(unity_Xx, unity_Yx)
    print(w1)

    # w1 = method1(Xtilde_tr, ytr)
    # w2 = method2(Xtilde_tr, ytr)
    # w3 = method3(Xtilde_tr, ytr)


    # # Report fMSE cost using each of the three learned weight vectors

    # fmse_tr_method1 = fMSE(w1, Xtilde_tr, ytr)
    # fmse_te_method1 = fMSE(w1, Xtilde_te, yte)

    # fmse_tr_method2 = fMSE(w2, Xtilde_tr, ytr)
    # fmse_te_method2 = fMSE(w2, Xtilde_te, yte)

    # fmse_tr_method3 = fMSE(w3, Xtilde_tr, ytr)
    # fmse_te_method3 = fMSE(w3, Xtilde_te, yte)

    # print("Method 1: Analytical Solution")
    # print("Training half-MSE: " + str(fmse_tr_method1))
    # print("Testing half-MSE: " + str(fmse_te_method1))

    # print("Method 2: Gradient Descent without Regularization")
    # print("Training half-MSE: " + str(fmse_tr_method2))
    # print("Testing half-MSE: " + str(fmse_te_method2))

    # print("Method 3: Gradient Descent with Regularization)")
    # print("Training half-MSE: " + str(fmse_tr_method3))
    # print("Testing half-MSE: " + str(fmse_te_method3))
    
    # # Visualizing Weights
    # vis_weights = [w1, w2, w3]
    # for i in range(3):
    #     image = vis_weights[i]
    #     image = np.delete(image, -1)
    #     image = image.reshape(48,48)
    #     plt.imshow(image)
    #     plt.show()

    # # Predicting on the test dataset
    # count = 0
    # rigorous_im = []
    # for i in range(len(yte)):
    #     image = Xtilde_te[:,i]
    #     pred = image*w3
    #     pred = np.sum(pred)
    #     if(abs(pred-yte[i]) > 40):
    #         rigorous_im.append(i)
    #         count += 1
    #         print(i, pred, yte[i])
    #     if count > 5:
    #         break
