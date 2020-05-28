# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

from __future__ import division
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

# YOUR CODE GOES HERE -- your PCA function
def your_PCA(X, K):
    W0 = np.random.randint(1,100,(1,150))
    W = W0
    for i in np.arange(100):
        C = X.dot(W.T).dot(np.linalg.pinv(W.dot(W.T)))
        W = np.linalg.pinv((C.T).dot(C)).dot(C.T).dot(X)

    return C, W

# plot everything
def plot_results(X, C):

    # Print points and pcs
    fig = plt.figure(facecolor = 'white',figsize = (10,4))
    ax1 = fig.add_subplot(121)
    for j in np.arange(0,n):
        plt.scatter(X[0][:].tolist(),X[1][:].tolist(),color = 'lime',edgecolor = 'k')

    s = np.arange(-np.abs(C[0,0]*(1e4)),np.abs(C[0,0]*(1e4)),.001)
    m = C[1,0]/C[0,0]
    ax1.plot(s, m*s, color = 'k', linewidth = 2)
    ax1.set_xlim(-.5, .5)
    ax1.set_ylim(-.5, .5)
    ax1.axis('off')

    # Plot projected data
    ax2 = fig.add_subplot(122)
    X_proj = np.dot(C, np.linalg.solve(np.dot(C.T,C),np.dot(C.T,X)))
    for j in np.arange(0,n):
        plt.scatter(X_proj[0][:].tolist(),X_proj[1][:].tolist(),color = 'lime',edgecolor = 'k')

    ax2.set_xlim(-.5, .5)
    ax2.set_ylim(-.5, .5)
    ax2.axis('off')

    return

# load in data
X = np.matrix(np.genfromtxt('C:/Users/10448/Desktop/9_2/PCA_demo_data.csv', delimiter=','))
n = np.shape(X)[0]
means = np.matlib.repmat(np.mean(X,0), n, 1)
X = X - means  # center the data
X = X.T
K = 1


# run PCA    
C, W = your_PCA(X, K)

# plot results
plot_results(X, C)
plt.show()

