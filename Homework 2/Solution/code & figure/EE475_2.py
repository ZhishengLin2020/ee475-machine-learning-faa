import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

# plus x and y index in csv file
# import bacteria_data.csv
df = pd.read_csv("./bacteria_data.csv")
print(df)

# change data frame to array
x = df["x"].to_numpy()
print(x)
y = df["y"].to_numpy()
print(y)

# add 1 colum of ones
X = np.vstack([x, np.ones(len(x))]).T
print(X)

# get Y
Y = np.log(y/(1-y))
print(Y)

# get the inverse of xp*xpT and xpyp and omega
xpxp_inv = np.linalg.inv(X.T.dot(X))
print(xpxp_inv)
xpyp = X.T.dot(Y)
print(xpyp)
omega = xpxp_inv.dot(xpyp.T)
print(omega)

# get b and w from omega
b = omega[1]
w = omega[0]

# write the equation and plot it
_ = plt.plot(x, y, 'o', label='Original data', markersize=10)
_ = plt.plot(x, 1/(1+np.exp(- w*x - b)), 'r', label='Fitted line')
_ = plt.legend()
plt.show()

