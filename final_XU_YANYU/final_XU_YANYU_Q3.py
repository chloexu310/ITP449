#Yanyu Xu
#ITP_449, Spring 2020
#Final Exam
#Question 3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree

#1. Create a DataFrame “ccDefaults” to store the credit card default data and set option to display all columns without any restrictions on the number of columns displayed. (1 point)
ccDefaults = pd.read_csv("ccDefaults.csv")
pd.set_option('display.max_columns', None)
df = pd.DataFrame(ccDefaults)

#2. Determine number of non-null samples and feature types. (1 point)
print(ccDefaults.info())

#3. Display the first 5 rows of ccDefaults. (1 point)
print(ccDefaults.head())

#4. Determine the dimensions of ccDefaults. (1 point)
print(ccDefaults.shape)

#5. Drop the ‘ID’ column from ccDefaults. (1 point)
ccDefaults.drop(columns='ID')

#6. Drop duplicates records from ccDefaults and identify if any duplicate records are dropped by printing out the dimensions of ccDefaults. (2 points)
ccDefaults.drop_duplicates(keep="first", inplace=True)
print(ccDefaults.shape)

#7. Drop records with null values from ccDefaults and identify if any null records are dropped by printing out the dimensions of ccDefaults. (2 points)
print(ccDefaults.isnull().sum())
ccDefaults.dropna(axis=0, inplace=True)
print(ccDefaults.shape)

#8. Display the correlation between the target variable “dpnm” and the remaining variables. (2 points)
print(ccDefaults.corr()['dpnm'].sort_values)

#9. Provide an analysis of the relationship between the target variable and the remaining variables. (2 points)
#The target variable has little relationship with most of the variables. Only the Pay_1 to Pay_6 has some positive
#relationship with the target variable

#10. Create a Feature Matrix, including only the 4 most important variables, and the Target Vector. (3 points)
X = ccDefaults[['PAY_1', 'PAY_2', 'PAY_3','PAY_4']]
y = ccDefaults['dpnm']

#11. Standardize the attributes of Feature Matrix. (1 point)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=['PAY_1', 'PAY_2', 'PAY_3','PAY_4'])
print(X.head())
print(X_scaled.head())

#12. Develop Decision Tree Classifier models with criterion of ‘entropy’ and ‘gini’ as well as max_depth ranging from 2 to 50. Use cross validation with 10 folds to determine the mean accuracy score for each of the models. (3 points)
cv_scores = []

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

for i in range(2, 50):
    for c in ['entropy', 'gini']:
        dtc = DecisionTreeClassifier(criterion=c,
                                     max_depth=i)
        scores = cross_val_score(dtc, X_scaled, y, cv=10, scoring='accuracy')
        mean = scores.mean()

        cv_scores.append([c, i, mean])

cv_scores = pd.DataFrame(cv_scores, columns=['Criterion',
                                             'Max Depth',
                                             'Accuracy'])
print(cv_scores)

#13. Plot 2 lineplots in the same figure depicting the values for max_depth and its corresponding mean accuracy scores for the models with entropy and gini criterion. Include a title, axes titles and a legend. Provide a screenshot of your plot. (4 points)
import matplotlib.pyplot as plt

plt.plot(cv_scores.loc[cv_scores['Criterion'] == 'entropy', 'Max Depth'],
         cv_scores.loc[cv_scores['Criterion'] == 'entropy', 'Accuracy'],
         color='m', label='Entropy')

plt.plot(cv_scores.loc[cv_scores['Criterion'] == 'gini', 'Max Depth'],
         cv_scores.loc[cv_scores['Criterion'] == 'gini', 'Accuracy'],
         color='g', label='Gini')

plt.title('Cross Validated Accuracy Score vs. Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy Score')
plt.legend()
plt.show()

#14. Overall, models with which criterion perform best? (1 point)
# criterion = 'entropy'

#15. Determine the best value of max_depth and criterion which maximizes the mean accuracy score and minimizes the computation time. (1 point)
print(cv_scores[cv_scores['Accuracy'] == cv_scores['Accuracy'].max()])


#16. Split the Feature Matrix and Target Vector into training and testing sets, reserving 20% of the data for testing. Set the ‘random_state’ parameter to 0 so that your results may be replicated. (2 points)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

#17. Develop a Decision Tree Classifier model with criterion and max_depth values from step 15. (3 points)
bestDTC = DecisionTreeClassifier(criterion='entropy',
                                 max_depth=2)
bestDTC.fit(X_train, y_train)

y_pred = bestDTC.predict(X_test)


#18. Display the accuracy of the model as well as the confusion matrix. (2 points)
print(bestDTC.score(X_test, y_test))
print(pd.crosstab(y_pred, y_test, rownames=['Prediction'], colnames=['Actual']))

#19. Plot the model and save as an image file. Walk through a single branch of the decision tree until you reach a leaf node and make a prediction, explaining the rules and branching. (4 points)
fn = X.columns
cn = ['3', '4', '5', '6', '7', '8']

plt.figure(figsize=(100, 100))
a = tree.plot_tree(bestDTC,
                   feature_names=fn,
                   class_names=cn)

plt.savefig('FinalDT.png')

#if the condition can be true, we will go to left branch, if not, we go to the right. We keep go down.

#20. Explain the difference between decision trees and random forests. Include the exact process of prediction that’s used in the Random Forest classifier model and how Decision Tree classifier models are incorporated. (3 points)
#A random forest is essentially a collection of Decision Trees. A decision tree is built on an entire dataset, using all the features/variables of interest, whereas a random forest randomly selects observations/rows and specific features/variables to build multiple decision trees from and then averages the results.