import numpy as np
import matplotlib.pyplot as plt
import csv

# sigmoid for softmax/logistic regression minimization
def sigmoid(z): 
    y = 1/(1+np.exp(-z))
    return y
    
# import training data 
def load_data(csvname):
    # load in data
    reader = csv.reader(open(csvname, "r"), delimiter=",")
    d = list(reader)

    # import data and reshape appropriately
    data = np.array(d).astype("float")
    X = data[:,0:8]
    y = data[:,8]
    y.shape = (len(y),1)
    
    # pad data with ones for more compact gradient computation
    o = np.ones((np.shape(X)[0],1))
    X = np.concatenate((o,X),axis = 1)
    X = X.T
    
    return X,y

# create a newton's method function for softmax and squared margin
def softmax_squared_newton(X,y):

    # define initial w for softmax and squared margin
    w_soft = (np.random.randn(9,1))/35
    w_squared = w_soft

    # define some common parameters
    X = X/35
    y = y
    max_its = 10

    # define parameters of softmax
    grad_soft = 1
    ite_soft = 0
    it_soft = []
    misc_soft = []

    # define parameters of squared margin
    grad_squared = 1
    ite_squared = 0
    it_squared = []
    misc_squared = []
    
    # softmax iteration
    while np.linalg.norm(grad_soft) > 10**(-12) and ite_soft < max_its:
        # calculate gradient
        r_0 = np.dot(X.T,w_soft)
        r_1 = -y*r_0
        r_2 = -sigmoid(r_1)
        r_3 = r_2*y
        r = r_3
        grad_soft = np.dot(X,r)
        
        # calculate misclassfication
        t_0 = np.dot(X.T,w_soft)
        t_1 = -y*t_0
        misclass_soft = np.sign(t_1)
        misclass_soft_new = (misclass_soft > 0).sum()
        misc_soft.append(misclass_soft_new)

        # calculate hessian
        t_2 = sigmoid(t_1)
        t_3 = 1 - t_2
        t_4 = t_2*t_3
        t_5 = t_4*(X.T)
        grad2_soft = np.dot(t_5.T,X.T)

        # calculate new w
        w_soft = w_soft - np.linalg.pinv(grad2_soft).dot(grad_soft)

        # calculate iteration
        ite_soft += 1
        it_soft.append(ite_soft)

    # squared margin iteration
    while np.linalg.norm(grad_squared) > 10**(-12) and ite_squared < max_its:
        # calculate gradient
        e_0 = np.dot(X.T,w_squared)
        e_1 = -y*e_0
        e_2 = (1+e_1)>0
        e_3 = e_2*(1+e_1)
        e = e_2*y
        grad_squared = -2*np.dot(X,e)
        
        # calculate misclassification
        f_0 = np.dot(X.T,w_squared)
        f_1 = -y*f_0
        misclass_squared = np.sign(e_1)
        misclass_squared_new = (misclass_squared > 0).sum()
        misc_squared.append(misclass_squared_new)

        # calculate hessian
        omega = e_1 > -1
        X_new = (X.T)*omega
        X_new = X_new.T
        grad2_squared = 2 * np.dot(X_new,X_new.T)

        # calculate new w
        w_squared = w_squared - np.linalg.pinv(grad2_squared).dot(grad_squared)

        # calculate iteration
        ite_squared += 1
        it_squared.append(ite_squared)

    return it_soft,misc_soft,it_squared,misc_squared



# plots everything 
def plot_all(a_ite,a_misclass,b_ite,b_misclass):
    
    plt.figure(dpi=110,facecolor='w')
    plt.plot(a_ite[1:],a_misclass[1:])
    plt.plot(b_ite[1:],b_misclass[1:])
    plt.title('breast cancer dataset')
    plt.xlabel('iteration')
    plt.ylabel('number of misclassification')
    plt.legend([r'$softmax\:cost$',r'$squared\:margin\:perceptron$'])
    plt.grid(True)
    plt.show()

# load in data
X,y = load_data('breast_cancer_data.csv')

# get the iteration and misclassification of softmax
a = softmax_squared_newton(X,y)
a_ite = a[0]
a_misclass = a[1]

# get the iteration and misclassification of squared margin
b = softmax_squared_newton(X,y)
b_ite = b[2]
b_misclass = b[3]


# plot points and separator
plot_all(a_ite,a_misclass,b_ite,b_misclass)