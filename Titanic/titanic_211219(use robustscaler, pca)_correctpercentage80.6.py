# Improving Accuracy of Titanic Project

# Strategy : ignore Name, Ticket, Cabin Columns and add data scaling and pca step

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train_df = pd.read_csv("C:/Users/user/Desktop/Machine Learning Practice/Titanic/train.csv")
test_df = pd.read_csv("C:/Users/user/Desktop/Machine Learning Practice/Titanic/test.csv")
submission_df = pd.read_csv("C:/Users/user/Desktop/Machine Learning Practice/Titanic/gender_submission.csv")

train_length = len(train_df)

df = pd.concat([train_df, test_df], ignore_index = True)

del df['Cabin']
del df['Ticket']
del df['Name']
del df['PassengerId']

df['Child'] = df['Age'] < 17
df['Adult'] = df['Age'] >= 17
df['Child'].replace(True, 1, inplace = True)
df['Child'].replace(False, 0, inplace = True)
df['Adult'].replace(True, 1, inplace = True)
df['Adult'].replace(False, 0, inplace = True)

# Filling NaN data
df['Age'].fillna(df.groupby(['Pclass', 'SibSp'])['Age'].transform('mean'), inplace = True)
df['Embarked'].fillna('S', inplace = True)
df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('mean'), inplace = True)

def merge_and_get(ldf, rdf, on, how = "inner", index = None):
    if index is True:
        return pd.merge(ldf, rdf, how = how, left_index = True, right_index = True)
    else:
        return pd.merge(ldf, rdf, how = how, on = on)

# log scale conversion
df = merge_and_get(df, np.log10(df['Fare'] + 0.5), on = None, index = True)
df.rename(columns = {'Fare_x' : 'Fare', 'Fare_y' : 'log_fare'}, inplace = True)

del df['Fare']

# make dummies for categorize

one_hot_df = merge_and_get(df, pd.get_dummies(df['Sex'], prefix = 'Sex'), on = None, index = True)
one_hot_df = merge_and_get(one_hot_df, pd.get_dummies(df['Embarked'], prefix = 'Embarked'), on = None, index = True)

del one_hot_df['Sex']
del one_hot_df['Embarked']

X_train = one_hot_df[:train_length][one_hot_df.columns[1:]] 
y_train = one_hot_df[:train_length][one_hot_df.columns[0]]
X_test = one_hot_df[train_length:][one_hot_df.columns[1:]] 
y_test = one_hot_df[train_length:][one_hot_df.columns[0]]

# Performing Standard Scaling
# transform data's scale

# 1st trial : StandardScaler
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
"""

# 2nd trial : Robust Scaler

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Performing PCA
"""
from sklearn.decomposition import PCA
pca = PCA(n_components = 12)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
"""

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators = 100, max_depth = 20, random_state = 0)

clf.fit(X_train, y_train)

Y_pre = clf.predict(X_test)

# importing model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#     training the model
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
y_pred = y_pred > 0.5

# importing model2
from sklearn import tree
clf2 = tree.DecisionTreeClassifier()
clf2 = clf2.fit(X_train, y_train)

y_pred2 = clf2.predict(X_test)
y_pred2 = y_pred2 > 0.5

correct = 0
wrong = 0
i = 0
for i in range(0, len(y_pred)):
    if submission_df['Survived'][i] == 0:
        if y_pred[i] == False:
            correct = correct + 1
        else:
            wrong = wrong + 1
    else:
        if y_pred[i] == True:
            correct = correct + 1
        else:
            wrong = wrong + 1

print("correct : {}, correct percentage : {}" .format(correct, correct/(correct + wrong)))

correct = 0
wrong = 0
i = 0
for i in range(0, len(y_pred)):
    if submission_df['Survived'][i] == 0:
        if y_pred2[i] == False:
            correct = correct + 1
        else:
            wrong = wrong + 1
    else:
        if y_pred2[i] == True:
            correct = correct + 1
        else:
            wrong = wrong + 1

print("correct : {}, correct percentage : {}" .format(correct, correct/(correct + wrong)))

"""
correct = 0
wrong = 0
i = 0
for i in range(0, len(y_pred)):
    if submission_df['Survived'][i] == 0:
        if Y_pre[i] == 0:
            correct = correct + 1
        else:
            wrong = wrong + 1
    else:
        if Y_pre[i] == 1:
            correct = correct + 1
        else:
            wrong = wrong + 1

print("correct : {}, correct percentage : {}" .format(correct, correct/(correct + wrong)))
"""
"""
y_result = pd.DataFrame(y_pred)

y_result.columns = ["Survived"]

y_result['Survived'].replace(True, 1, inplace = True)
y_result['Survived'].replace(False, 0, inplace = True)

temp = submission_df['PassengerId']

submission_file = pd.concat([temp, y_result], axis = 1)

submission_file.to_csv("C:/Users/user/Desktop/Machine Learning Practice/Titanic/submit211221.csv", index = False)
"""