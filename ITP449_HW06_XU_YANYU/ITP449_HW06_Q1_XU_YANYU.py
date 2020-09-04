# Yanyu Xu
# ITP 449 Spring 2020
# HW06
# Question 1
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


#Create a Support Vector Machine (SVM) model for diabetes prediction:
#1. Create a DataFrame “diabetes_svm” to store the diabetes data and set option to display all columns without any restrictions on the number of columns displayed.
diabetes_svm = pd.read_csv('diabetes.csv')
pd.set_option('display.max_columns', None)
df = pd.DataFrame(diabetes_svm)
print(df.info())

#2. Determine the dimensions of the DataFrame.
print(diabetes_svm.shape)

#3. Determine the number of non-null samples and variable types.
print(diabetes_svm.notnull().any())

#4. Display the first 5 rows of the “diabetes_svm” DataFrame.
print(diabetes_svm.head())

#5. Print summary statistics (i.e. count, mean, ... 75%, max)
print(diabetes_svm.describe())

#6. Provide an analysis of what needs to be addressed in terms of data wrangling. Note that there are two categories of data wrangling to consider.
wrangling = diabetes_svm.dropna(axis=0)
print(wrangling)
# There is no null data and no duplicates

#7. Compute a correlation matrix depicting the relationship between attributes.
correlation = diabetes_svm.corr()
print("The correlation is", correlation)

#8. Determine the number of diabetes positive and diabetes negative cases in the provided dataset.
print("The number of diabetes positive is", sum(df["Outcome"] == 1))
print("The number of diabetes negative is", sum(df["Outcome"] == 0))

#9. Display the relationship between all variables in the dataset using pairplot() function from the Seaborn package.
sb.pairplot(data=diabetes_svm)
plt.show()

#10. Describe the relationship between the target variable （outcome) and other variables.
#I see the strongest relationship between outcome and other variables is with Glucose. It have 0.46 correlation number.
#Also, for other factors such as Pregnancies, BMI and Age all over an over 0.2 correlation number. It can show a semi-strong relationship.
#For DiabetesPedigreeFunction and Insulin, they have a slightly weak relationship comparing to other factors, but they are still above 0.1
#The weakest relationship with outcome are SkinThickness and BloodPressure, which they have a lower than 0.1 correlation ratio

#11. Create the Feature Matrix and Target Vector, using your analysis above to inform which variables to include in the Feature Matrix.
X = diabetes_svm[["Pregnancies","Glucose","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
y = diabetes_svm["Outcome"]

print(X.shape)
print(y.shape)

#12. Scale the variables in the Feature Matrix.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#13. Explain why it is important to scale these variables.
#To scale these variables is to standardlize the values

#14. Develop a Support Vector Machine (SVM) model with kernel type ‘linear’ with default values for parameters C and Gamma.
model = svm.SVC(kernel = 'linear')
X = diabetes_svm[["Pregnancies","Glucose","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
y = diabetes_svm["Outcome"]

#15. Use 10 fold cross validation to determine the mean accuracy score of the above model.
accuracies = cross_val_score(model,X_scaled,y,cv=10,scoring="accuracy")
mean = accuracies.mean()
print(mean)


#16. Develop a Support Vector Machine (SVM) model with kernel type ‘rbf’ with default values for parameters C and Gamma.
model1 = svm.SVC(kernel = 'rbf')

#17. Use 10 fold cross validation to determine the mean accuracy score of the above model.
accuracies1 = cross_val_score(model,X_scaled,y,cv=10,scoring="accuracy")
mean1 = accuracies1.mean()
print(mean1)

#18. Comparing the accuracy scores for the ‘linear’ vs. ‘rbf’ models, select the model with the higher accuracy for the remainder of the steps.
#from the model, rbf is better.

#19. Split the scaled Feature Matrix and Target Vector into training and testing sets, reserving 25% of the data for testing.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size= 0.25)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

print(len(y_test))

#20. Develop SVM models with the kernel selected in step 17 and all combinations of C values [0.001, 0.01, 0.1, 1, 10, 100, 1000] and Gamma values [0.001, 0.01, 0.1, 1, 10, 100, 1000], computing the accuracy score for each model.
C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
Gamma = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
max = 0
bestc = 0
bestg = 0
cv_scores = []
for c in C:
    for g in Gamma:
        model = svm.SVC(kernel="rbf",gamma=g,C=c)
        model.fit(X_train,y_train)
        score = model.score(X_test,y_test)
        print("C:", c, "Gamma:", g, "Score:", cv_scores)
        if score > max:
            max = score
            bestc = c
            bestg = g

print(max,bestc,bestg)

#21. Select the model with C and Gamma values that maximizes the accuracy score.
model = svm.SVC(kernel="rbf", gamma=bestg, C=bestc)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

#22. Print the confusion matrix for the above model. Describe what each value means.
print(pd.crosstab(y_pred,y_test))

#The upper left is showing our prediction that the person do not have diabetes, and the person really do not have diabetes
#The upper right is showing our prediction that the person do not have diabetes, and the person has diabetes
#The lower left is showing our prediction that the person has diabetes, and the person do not have diabetes
#The lower right is showing our prediction that the person has diabetes, and the person really has diabetes

#23. In this case, which errors (false positives or false negatives) would be important to minimize? Explain why.
#The false negative will be more important because if we have a false data, the sick people will not be able to get cured.