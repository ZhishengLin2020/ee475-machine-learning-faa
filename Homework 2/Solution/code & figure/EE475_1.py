import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

# plus time and loan index in csv file
# import student_debt.csv
df = pd.read_csv("./student_debt.csv")
print(df)

# change data frame to array
x = df["year"].to_numpy()
print(x)
y = df["loan"].to_numpy()
print(y)

# write the equation and plot it
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y, rcond=None)[0]
_ = plt.plot(x, y, 'o', label='Original data', markersize=10)
_ = plt.plot(x, m*x + c, 'r', label='Fitted line')
_ = plt.legend()

# predict the student debt in 2050
y_2050 = m*2050 + c
print(y_2050)

plt.show()