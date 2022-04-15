import numpy as np
import matplotlib.pyplot as plt
from cvxopt import solvers, matrix
from homework4_Gandhi import SVM4342

def get_hyperplane(X, w):
    b = -1*X.T.dot(w)

    X = X.T
    ones = np.ones((X.shape[0],1))
    Xtilde = np.append(X, ones, axis=1)
    Wtilde = np.append(w, 1)

    hyperplane = Xtilde.dot(Wtilde)
    
    print(hyperplane)
    return hyperplane 

if __name__ == "__main__":
    X = np.load("hw4_X.npy")
    y = np.load("hw4_y.npy")
    n = X.shape[1]//2

    # x = np.arange(-8, +8, 0.01)

    plt.scatter(X[0,0:n], X[1,0:n])
    plt.scatter(X[0,n:], X[1,n:])

    w1 = np.array([0,1])
    w2 = np.array([-0.3,1])
    h1 = get_hyperplane(X,w1)
    h2 = get_hyperplane(X,w2)
    
    svmClassifier = SVM4342()
    svmClassifier.fit(X.T, y)

    # Plot some arbitrary parallel lines (*not* separating hyperplanes) just for an example
    plt.plot(X, X*-1.9+3, 'k-')
    plt.plot(X, X*-1.9+3+1, 'k--')
    plt.plot(X, X*-1.9+3-1, 'k:')

    plt.show()

    