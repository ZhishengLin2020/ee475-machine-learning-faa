# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

from __future__ import division
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt


# YOUR CODE GOES HERE
def your_K_means(X, K, C0):
    C = C0
    for i in np.arange(100):
        k_star_arr = []
        W = np.zeros([K,np.shape(X)[1]])
        for p in np.arange(np.shape(X)[1]):
            for k in np.arange(K):
                k_s = (np.linalg.norm(C[:,k].reshape(2,1)-X[:,p].reshape(2,1),ord=2))**2
                k_star_arr.append(k_s)
            k_star = np.argmin(k_star_arr)
            k_star_arr = []
            if k_star == 0:
                W[:,p] = np.array([[1,0]])
            else:
                W[:,p] = np.array([[0,1]])
        Xp_sum = X.dot(W.T)
        Sk_inv = np.linalg.inv(W.dot(W.T))
        C = Sk_inv.dot(Xp_sum)

    return C, W

def plot_results(X, C, W, C0):

    K = np.shape(C)[1]

    # plot original data
    fig = plt.figure(facecolor = 'white')
    ax1 = fig.add_subplot(121)
    plt.scatter(X[0,:],X[1,:], s = 50, facecolors = 'k')
    plt.title('original data')
    ax1.set_xlim(-.55, .55)
    ax1.set_ylim(-.55, .55)
    ax1.set_aspect('equal')

    plt.scatter(C0[0,0],C0[1,0],s = 100, marker=(5, 2), facecolors = 'b')
    plt.scatter(C0[0,1],C0[1,1],s = 100, marker=(5, 2), facecolors = 'r')

    # plot clustered data
    ax2 = fig.add_subplot(122)
    colors = ['b','r']

    for k in np.arange(0,K):
        ind = np.nonzero(W[k][:]==1)[0]
        plt.scatter(X[0,ind],X[1,ind],s = 50, facecolors = colors[k])
        plt.scatter(C[0,k],C[1,k], s = 100, marker=(5, 2), facecolors = colors[k])

    plt.title('clustered data')
    ax2.set_xlim(-.55, .55)
    ax2.set_ylim(-.55, .55)
    ax2.set_aspect('equal')
    
# load data
X = np.array(np.genfromtxt('C:/Users/10448/Desktop/9_1/Kmeans_demo_data.csv', delimiter=','))

C0 = np.array([[0,0],[0,0.5]])   # initial centroid locations

# run K-means
K = np.shape(C0)[1]

C, W = your_K_means(X, K, C0)

# plot results
plot_results(X, C, W, C0)
plt.show()