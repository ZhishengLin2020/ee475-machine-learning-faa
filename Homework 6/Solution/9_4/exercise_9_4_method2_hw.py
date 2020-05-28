# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

from __future__ import division
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt


# YOUR CODE GOES HERE - recommender systems via matrix completion
def matrix_complete(X, K):
    C0 = np.random.randint(100,200,(100,5))
    C = C0
    W_arr = np.zeros((5,200))
    C_arr = np.zeros((100,5))
    for i in np.arange(60):
        for p in np.arange(np.shape(X)[1]):
            wp = ((C.T).dot(X[:,p].reshape(100,1)))/(((C).dot(C.T)).sum())
            W_arr[:,p] = np.squeeze(wp)
            if X[:,p].sum() == 0:
                wp = np.array([0,0,0,0,0])
                W_arr[:,p] = wp
        W = W_arr
        for n in np.arange(np.shape(X)[0]):
            cn = (X[n,:].dot(W.T)).dot(np.linalg.pinv(W.dot(W.T)))
            C_arr[n,:] = np.squeeze(cn)
            if X[n,:].sum() == 0:
                cn = np.array([0,0,0,0,0])
                C_arr[n,:] = cn
        C = C_arr
        W_arr = np.zeros((5,200))
        C_arr = np.zeros((100,5))

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