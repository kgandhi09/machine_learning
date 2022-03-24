from matplotlib.pyplot import axis
import numpy as np

def problem1 (A, B):
    return A + B

def problem2 (A, B, C):
    return np.dot(A,B) - C

def problem3 (A, B, C):
    return np.multiply(A,B) - np.transpose(C)

def problem4 (x, S, y):
    return np.dot(np.dot(np.transpose(x), S), y) 

def problem5 (A):
    return np.ones(A.shape[0])

def problem6 (A):
    B = A
    return np.fill_diagonal(B, 0) 

def problem7 (A, alpha):
    I = np.eye(A.shape[0])
    return A + alpha*I

def problem8 (A, i, j):
    return A[j,i]

def problem9 (A, i):
    return np.sum(A[i])

def problem10 (A, c, d):
    return A.mean(where=c<A<d)

def problem11 (A, k):
    w,v = np.linalg.eig(A)
    id = w.argsort()[::-1]
    v = v[:,id]
    return v[:,:k]

def problem12 (A, x):
    return np.linalg.solve(A,x)

def problem13 (x, k):
    return np.repeat(x, k, axis=0) 

def problem14 (A):
    return np.random.permutation(A, axis=0)
