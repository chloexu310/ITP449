# Yanyu Xu
# ITP 449 Spring 2020
# HW06
# Question 2
import os
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

#Create a K-Nearest Neighbor (KNN) model for diabetes prediction:
#1. Create a DataFrame “diabetes_knn” to store the diabetes data and set option to display all columns without any restrictions on the number of columns displayed.
diabetes_knn = pd.read_csv('diabetes.csv')
pd.set_option('display.max_columns', None)
df = pd.DataFrame(diabetes_knn)
print(df.info())
#2. Repeat steps 2-12 from Question 1. Use the scaled Feature Matrix and Target Vector for the remainder of the steps.
#1-2
print(diabetes_knn.shape)
#1-3
print(diabetes_knn.notnull().any())
#1-4
print(diabetes_knn.head())
#1-5
print(diabetes_knn.describe())
#1-6
wrangling = diabetes_knn.dropna(axis=0)
print(wrangling)
# There is no null data and no duplicates
#1-7
correlation = diabetes_knn.corr()
print("The correlation is", correlation)
#1-8
print("The number of diabetes positive is", len(df["Outcome"] == 1))
print("The number of diabetes negative is", len(df["Outcome"] == 0))

#1-9
sb.pairplot(data=diabetes_knn)
plt.show()

#1-10
#I see the strongest relationship between outcome and other variables is with Glucose. It have 0.46 correlation number.
#Also, for other factors such as Pregnancies, BMI and Age all over an over 0.2 correlation number. It can show a semi-strong relationship.
#For DiabetesPedigreeFunction and Insulin, they have a slightly weak relationship comparing to other factors, but they are still above 0.1
#The weakest relationship with outcome are SkinThickness and BloodPressure, which they have a lower than 0.1 correlation ratio

#1-11
X = diabetes_knn[["Pregnancies","Glucose","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
y = diabetes_knn["Outcome"]
print(X.shape)
print(y.shape)

#1-12
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#3. Use 10 fold cross validation to determine the mean accuracy scores for K-Nearest Neighbor models with k values (number of neighbors) ranging from 1 to the number of training observations.
cv_scores = []
folds = 10
ks = list(range(1, int(len(X_scaled)*((folds-1)/folds))))
for i in ks:
    if i % 2 == 0:
        ks.remove(i)
max = 0
best = 0
for k in ks:
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model,X_scaled,y,cv=10,scoring="accuracy")
    mean = scores.mean()
    print(mean)
    cv_scores.append(mean)
    if (mean > max):
        max = mean
        best = k


#4. Display a line plot with the k values on the X axis and mean accuracy scores on the Y axis.
plt.plot(ks, cv_scores, color = 'm', linestyle ='dashed', marker='o')
plt.title('Cross Validated Mean Accuracy Scores vs. K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Accuracy Scores')
plt.show()

#5. Which value of k provides the highest accuracy?
print(best)

#6. Split the scaled Feature Matrix and Target Vector into training and testing sets, reserving 25% of the data for testing.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size= 0.25)


#7. Develop a K-Nearest Neighbor (KNN) model using the k value from step 5.
classifier = KNeighborsClassifier(n_neighbors=best)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#8. Print the confusion matrix for the above model. Describe what each value means.
print(pd.crosstab(y_pred,y_test))

#The upper left is showing our prediction that the person do not have diabetes, and the person really do not have diabetes
#The upper right is showing our prediction that the person do not have diabetes, and the person has diabetes
#The lower left is showing our prediction that the person has diabetes, and the person do not have diabetes
#The lower right is showing our prediction that the person has diabetes, and the person really has diabetes

#9. Compare this KNN model confusion matrix to the SVM confusion matrix from Question 1. Provide an analysis.
#When we compare it, the true negative and true positive decreased and the false negative and false positive increased.

#10. Which model is better? Why?
#I think the SVM model is better because we can see there is a more correct accuracies for SVM and lower error in SVM.