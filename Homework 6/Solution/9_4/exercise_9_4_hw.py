# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

from __future__ import division
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt


# YOUR CODE GOES HERE - recommender systems via matrix completion
def matrix_complete(X, K):
    C0 = np.random.randint(1,200,(100,5))
    W_arr = []
    C_arr = []
    X_sum = 0
    C_sum = 0
    W_sum = np.zeros((5,5))
    XW_sum = 0
    C = C0
    for m in np.arange(100):
        for p in np.arange(200):
            for i in np.arange(100):
                if X[:,p].sum() == 0:
                    w = np.array([0,0,0,0,0]).reshape(5,1)
                    W_arr.append(w)
                    break
                if X[i,p] != 0:
                    C_sum = C_sum+(C[i,:].reshape(1,5)).dot(C[i,:].reshape(5,1))
                    X_sum = X_sum+X[i,p]*(C[i,:].reshape(5,1))
            w = X_sum/C_sum[0,0]
            W_arr.append(w)
        W = np.squeeze(W_arr).T
        W = np.delete(W,-1,axis=1)
        W = np.nan_to_num(W)
        for n in np.arange(100):
            for j in np.arange(200):
                if X[n,:].sum() == 0:
                    c = np.array([0,0,0,0,0]).reshape(1,5)
                    C_arr.append(c)
                    break
                if X[n,j] != 0:
                    W_sum = W_sum+(W[:,j].reshape(5,1)).dot(W[:,j].reshape(1,5))
                    XW_sum = XW_sum+X[n,j]*(W[:,j].reshape(1,5))
            c = XW_sum.dot(np.linalg.pinv(W_sum,1e-2))
            C_arr.append(c)
        C = np.squeeze(C_arr)
        C = np.nan_to_num(C)
        W_arr = []
        C_arr = []
        X_sum = 0
        C_sum = 0
        W_sum = np.zeros((5,5))
        XW_sum = 0

    return C, W

def plot_results(X, X_corrupt, C, W):

    gaps_x = np.arange(0,np.shape(X)[1])
    gaps_y = np.arange(0,np.shape(X)[0])

    # plot original matrix
    fig = plt.figure(facecolor = 'white',figsize = (30,10))
    ax1 = fig.add_subplot(131)
    plt.imshow(X,cmap = 'hot',vmin=0, vmax=20)
    plt.title('original')

    # plot corrupted matrix
    ax2 = fig.add_subplot(132)
    plt.imshow(X_corrupt,cmap = 'hot',vmin=0, vmax=20)
    plt.title('corrupted')

    # plot reconstructed matrix
    ax3 = fig.add_subplot(133)
    recon = np.dot(C,W)
    plt.imshow(recon,cmap = 'hot',vmin=0, vmax=20)
    RMSE_mat = np.sqrt(np.linalg.norm(recon - X,'fro')/np.size(X))
    title = 'RMSE-ALS = ' + str(RMSE_mat)
    plt.title(title,fontsize=10)
    
# load in data
X = np.array(np.genfromtxt('C:/Users/10448/Desktop/9_4/recommender_demo_data_true_matrix.csv', delimiter=','))
X_corrupt = np.array(np.genfromtxt('C:/Users/10448/Desktop/9_4/recommender_demo_data_dissolved_matrix.csv', delimiter=','))

K = np.linalg.matrix_rank(X)

# run ALS for matrix completion
C, W = matrix_complete(X_corrupt, K)

# plot results
plot_results(X, X_corrupt, C, W)
plt.show()