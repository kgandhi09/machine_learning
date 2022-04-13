import numpy as np
import matplotlib.pyplot as plt

X = np.load("hw4_X.npy")
y = np.load("hw4_y.npy")
n = X.shape[1]//2

x = np.arange(-8, +8, 0.01)
plt.scatter(X[0,0:n], X[1,0:n])
plt.scatter(X[0,n:], X[1,n:])

# Plot some arbitrary parallel lines (*not* separating hyperplanes) just for an example
plt.plot(x, x*-1.9+3, 'k-')
plt.plot(x, x*-1.9+3+1, 'k--')
plt.plot(x, x*-1.9+3-1, 'k:')
plt.show()
