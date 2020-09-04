#Yanyu Xu
#ITP_449, Spring 2020
#Final Exam
#Question 2

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#1. Create a DataFrame “bcDiagnosis” to store the breast cancer diagnosis data and set option to display all columns without any restrictions on the number of columns displayed. (1 point)
bcDiagnosis = pd.read_csv("breastCancer(1).csv")
pd.set_option('display.max_columns', None)
df = pd.DataFrame(bcDiagnosis)

#2. Determine number of non-null samples and feature types. (1 point)
print(bcDiagnosis.info())

#3. Display the first 5 rows of bcDiagnosis. (1 point)
print(bcDiagnosis.head())

#4. Determine the dimensions of bcDiagnosis. (1 point)
print(bcDiagnosis.shape)

#5. Drop duplicates records from bcDiagnosis and identify if any duplicate records are dropped by printing out the dimensions of bcDiagnosis. (2 points)
bcDiagnosis.drop_duplicates(keep="first", inplace=True)
print(bcDiagnosis.shape)

#6. Drop records with null values from bcDiagnosis and identify if any null records are dropped by printing out the dimensions of t bcDiagnosis. (2 points)
print(bcDiagnosis.isnull().sum())
bcDiagnosis.dropna(axis=0, inplace=True)
print(bcDiagnosis.shape)

#7. Print the summary statistics of bcDiagnosis. (1 point)
print(bcDiagnosis.describe())

#8. Drop the ‘id’ and ‘bare_nuceloli’ columns from bcDiagnosis. (1 point)
bcDiagnosis.drop(columns='id')
bcDiagnosis.drop(columns='bare_nucleoli')

#9. Display the correlation between the target variable ‘class’ and the remaining variables. (2 points)
print(bcDiagnosis.corr()['class'].sort_values)

#10. Create the Target Vector and the Feature Matrix that only includes variables that have a high correlation (>=|0.75|) with the target variable. (1 point)
X = bcDiagnosis[['size_uniformity','shape_uniformity','bland_chromatin']]
y = bcDiagnosis['class']

#11. Standardize the attributes of the Feature Matrix and comment why this is important. (2 points)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=['size_uniformity','shape_uniformity','bland_chromatin'])
print(X.head())
print(X_scaled.head())

# Because the original features are on different scales (different magnitudes for the range of values),
# variables with a higher ranged magnitude may be weighed more heavily in the model. Normalizing the
# features assures that all variables will be on an even playing field.


#12. Develop a SVM model with a “rbf” kernel and use cross validation with 10 folds to compute and print the mean accuracy of the model. (3 points)
model_rbf = svm.SVC(kernel='rbf')

scoresRBF = cross_val_score(model_rbf, X_scaled, y, cv=10, scoring='accuracy')
print(scoresRBF.mean())

#13. Develop a SVM model with a “linear” kernel and use cross validation with 10 folds to compute and print mean accuracy of the model. (3 points)

model_lin = svm.SVC(kernel='linear')

scoresLin = cross_val_score(model_lin, X_scaled, y, cv=10, scoring='accuracy')
print(scoresLin.mean())

#14. Which of the two kernels maximizes the mean accuracy score? (1 point)
# kernel = RBF

#15. Develop SVM models with the best performing kernel identified in step 14 and the following values of gamma: [0.001, 0.01, 0.1, 1, 10, 100, 1000]. Use cross validation with 10 folds to compute and print the mean accuracy scores. (3 points)
C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
Gamma = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
max1 = 0
bestc = 0
bestg = 0
cv_scores = []
for c in C:
    for g in Gamma:
        model = svm.SVC(kernel="rbf",gamma=g,C=c)
        model.fit(X,y)
        score = model.score(X,y)
        print("C:", c, "Gamma:", g, "Score:", cv_scores)
        if score > max1:
            max1 = score
            bestc = c
            bestg = g

print(max1,bestc,bestg)

#16. Which value of gamma maximizes the mean accuracy score? (1 point)


#17. Develop KNN models with values of k ranging from 1 to 120. Use cross validation with 10 folds to determine the mean accuracy scores. (3 points)

meanScoresKNN = []
ks = range(1, 120)

for k in ks:
    model_knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model_knn, X_scaled, y, cv=10, scoring='accuracy')
    mean = scores.mean()

    meanScoresKNN.append(mean)

#18. Plot a lineplot depicting the values for k and its corresponding mean accuracy score for the model. Include a title and axes labels. Provide a screenshot of your plot. (3 points)
plt.plot(ks, meanScoresKNN, color='m', linestyle='dashed', marker='o')

plt.title('Cross Validated Accuracy Score vs. K Value')
plt.xlabel('K Value')
plt.ylabel('Accuracy Score')
plt.show()


#19. Which value of k maximizes the mean accuracy score? (1 points)
optimalK = meanScoresKNN.index(max(meanScoresKNN)) + 1
print(optimalK)

#20. Split the Feature Matrix and Target Vector into training and testing sets, reserving 20% of the data for testing. Set the ‘random_state’ parameter to 0 so that your results may be replicated. (2 points)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

#21. Develop a SVM model using the optimal kernel and gamma and display the accuracy score of the model as well as the confusion matrix. (1 point)
bestSVM = svm.SVC(kernel='rbf')
bestSVM.fit(X_train, y_train)

y_pred = bestSVM.predict(X_test)

print(bestSVM.score(X_test, y_test))
print(pd.crosstab(y_pred, y_test, rownames=['Prediction'], colnames=['Actual']))
#22. Develop a KNN model using the optimal k value and display the accuracy score of the model as well as the confusion matrix. (1 point)
bestKNN = KNeighborsClassifier(n_neighbors=optimalK)
bestKNN.fit(X_train, y_train)

y_pred = bestKNN.predict(X_test)

print(bestKNN.score(X_test, y_test))
print(pd.crosstab(y_pred, y_test, rownames=['Prediction'], colnames=['Actual']))

#23. Which of the models, KNN or SVM, provides the highest accuracy and best predictions? (1 point)
# Although they are similar, the SVM model provides slightly better accuracy
# but it should be noted that KNN outperforms SVM for certain qualities

#24. Make a diagnosis prediction for a case with the attribute values set to the 75% values computed in step 7. (2 points)
a = pd.DataFrame(np.array([5,5,5]).reshape(1,3))
print(bestSVM.predict(a))