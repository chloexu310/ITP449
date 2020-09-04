def Q1():
    print("Multiplication Table\n")
    num = int(input("Please enter a whole number"))
    symbol = "X"
    for index in range(num):
        product = index * num
        msg = ""
        if index < 10:
            msg = " "
        msg = msg + str(index)
        print(msg, symbol, num, "=", product)
        print("Math is fun!")

# Q1()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import  LinearRegression
import numpy as np
def Q2():
    frame = pd.read_csv("../../../../../Desktop/sample midterm from last week/mtcars.csv")
    pd.set_option("display.max_columns", None)
    print(frame)
    frame.set_index("Car Name", inplace=True)
    print(frame.head())
    hp = frame["hp"]
    mpg = frame["mpg"]
    plt.scatter(hp,mpg,color="m", marker=".")
    plt.ylabel("Miles per Gallon")
    plt.xlabel("Horsepower")
    model = LinearRegression()
    x = hp[:,np.newaxis]
    y =mpg
    model.fit(x,y)
    print(model.intercept_)
    print(model.coef_)
    # 335
    maxNum = np.max(hp)
    newX = np.arange(50,maxNum+1)
    newX = newX[:,np.newaxis]
    y = model.predict(newX)
    plt.plot(newX,y,c="k")
    plt.show()


# Q2()

