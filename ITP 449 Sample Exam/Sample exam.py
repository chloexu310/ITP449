#similar to the homework

#Q3
import pandas as pd
frame = pd.read_csv('mtcars.csv')

#Q3-2 Print DataFrame
pd.set_option('max_columns', None)
print(frame)

#Q3-3 Add car name as index
frame.set_index('Cat Name', inplace=True) #inplace means update the frame so it runs,
# if you do not put it, it will not run
print(frame)

#03-4 Print frame

#03-5 Scatter plot
import matplotlib.pyplot as plt
plt.scatter(frame['hp'], frame['mpg'], c='m')
plt.xlabel('Horsepower')
plt.ylabel('Miles per Gallon')
plt.show()

#03-6 Linear Regression X and Y
from sklearn.linear_model import  LinearRegression


#Q3-7 Feature matrix and target vector
y = frame.mpg
X = frame.hp
print(X.shape)

import numpy as np
X = X[:, np.newaxis]
print(y.shape)
print(X.shape)

model.fit(X,y)
print(model.intercept_)
print(model.coef_)

plt.scatter(frame['hp'], frame['mpg'], c='m')

xCor = np.arange(50,350,20)
xCor = xCor[:, np.newaxis]
yCor = model.predict(xCor)

plt.plot(xCor,yCor, c = 'k')
plt.xlabel('Horsepower')
plt.ylabel('Miles per Gallon')
plt.show()