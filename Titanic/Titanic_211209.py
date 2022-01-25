# Titanic Machine Learning Project
# plot 제외 데이터 전처리만 모음

# import modules
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import seaborn as sns
import matplotlib.pyplot as plot
from collections import Counter

# read data from csv files
df_test = pd.read_csv("C:/Users/user/Desktop/Machine Learning Practice/Titanic/test.csv")
df_train = pd.read_csv("C:/Users/user/Desktop/Machine Learning Practice/Titanic/train.csv")

df = pd.concat([df_train, df_test], ignore_index = True)

# 결측치 처리

df['Age'].fillna(df.groupby('Pclass')['Age'].transform('mean'), inplace = True)

df['Embarked'].fillna('S', inplace = True)

# Category Data
# Sex, Pclass, Embarked data column을 get_dummies로 column을 새로 만들어 합병하는 식으로 data categorize함

def merge_and_get(ldf, rdf, on, how = "inner", index = None):
    if index is True:
        return pd.merge(ldf, rdf, how = how, left_index = True, right_index = True)
    else:
        return pd.merge(ldf, rdf, how = how, on = on)

object_columns = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
numeric_columns = ['Age', 'SibSp', 'Parch', 'Fare']

for col_name in object_columns:
    df[col_name] = df[col_name].astype(object)

for col_name in numeric_columns:
    df[col_name] = df[col_name].astype(float)

df['Parch'] = df['Parch'].astype(int)
df['SibSp'] = df['SibSp'].astype(int)

one_hot_df = merge_and_get(
    df, pd.get_dummies(df['Sex'], prefix = 'Sex'), on = None, index = True)
one_hot_df = merge_and_get(
    one_hot_df, pd.get_dummies(df['Pclass'], prefix = "Pclass"), on = None, index = True)
one_hot_df = merge_and_get(
    one_hot_df, pd.get_dummies(df['Embarked'], prefix = "Embarked"), on = None, index = True)


# Mean Encoding
# Mean Encoding 사용시 corr이 더 증가하는 것을 확인할 수 있음

temp_columns = ['Sex', 'Pclass', 'Embarked']

me_list = []

for col_name in temp_columns:
    temp_df = pd.merge(one_hot_df[col_name], df['Survived'], left_index = True, right_index = True)
    temp_df["me_"+col_name] = temp_df[col_name].replace(temp_df.groupby(col_name)['Survived'].mean())
    
    me_list.append(temp_df.drop("Survived", axis = 1))
    
temp_df = pd.merge(pd.concat(me_list, axis=1)[["me_" + col_name for col_name in temp_columns]], one_hot_df['Survived'], left_index = True, right_index = True)


# Categorical Combination

one_hot_df['Sex-Pclass'] = df['Sex'].map(str) + df['Pclass'].map(str)
one_hot_df['Embarked-Pclass'] = df['Embarked'].map(str) + df['Pclass'].map(str)

one_hot_df = merge_and_get(
    one_hot_df, pd.get_dummies(one_hot_df['Sex-Pclass'], prefix = 'SexPclass'), on = None, index = True)
one_hot_df = merge_and_get(
    one_hot_df, pd.get_dummies(one_hot_df['Embarked-Pclass'], prefix = 'EmbarkedPclass'), on = None, index = True)

# Log transformation
# 데이터의 분포가 극단적으로 모였을 때
# 선형 모델은 데이터가 정규분포때 적합하다.
one_hot_df['Fare'] = one_hot_df['Fare'].fillna(one_hot_df['Fare'].mean())
log_one_hot_df = merge_and_get(one_hot_df, np.log10(one_hot_df['Fare']+0.5), on = None, index = True)
log_one_hot_df.rename(columns = {'Fare_x' : 'Fare', 'Fare_y' : 'log_fare'}, inplace = True)

# String Handling
log_one_hot_df["is_mr"] = log_one_hot_df['Name'].str.lower().str.contains(pat = "mr.")
log_one_hot_df["is_miss"] = log_one_hot_df['Name'].str.lower().str.contains(pat = "miss.")
log_one_hot_df["is_mrs"] = log_one_hot_df['Name'].str.lower().str.contains(pat = "mrs.")

# Counter((log_one_hot_df['Ticket'].str.lower() + " ").sum().split()).most_common(30)
log_one_hot_df["is_pc"] = log_one_hot_df['Ticket'].str.lower().str.contains(pat = "pc")
log_one_hot_df["is_ca"] = log_one_hot_df['Ticket'].str.lower().str.contains(pat = "c.a.")
log_one_hot_df["is_paris"] = log_one_hot_df['Ticket'].str.lower().str.contains(pat = "paris")
log_one_hot_df["is_soton"] = log_one_hot_df['Ticket'].str.lower().str.contains(pat = "soton")
log_one_hot_df["is_ston"] = log_one_hot_df['Ticket'].str.lower().str.contains(pat = "ston")
log_one_hot_df["is_so"] = log_one_hot_df['Ticket'].str.lower().str.contains(pat = "s.o")


# cabin 결측치 처리
# Counter((test.str.lower() + " ").sum().split()).most_common(30)
test = log_one_hot_df['Cabin'].fillna("9999")

log_one_hot_df['is_cabin_none'] = test.str.contains(pat = '9999')
log_one_hot_df['is_cabin_a'] = test.str.contains(pat = 'a')
log_one_hot_df['is_cabin_b'] = test.str.contains(pat = 'b')
log_one_hot_df['is_cabin_c'] = test.str.contains(pat = 'c')
log_one_hot_df['is_cabin_d'] = test.str.contains(pat = 'd')
log_one_hot_df['is_cabin_e'] = test.str.contains(pat = 'e')
log_one_hot_df['is_cabin_f'] = test.str.contains(pat = 'f')
log_one_hot_df['is_cabin_g'] = test.str.contains(pat = 'g')

def count_cabin(x):
    if type(x) is int:
        return 0
    else:
        return len(x)

log_one_hot_df['number_of_cabin'] = log_one_hot_df['Cabin'].str.split(" ").fillna(0).map(count_cabin)
log_one_hot_df['log_number_of_cabin'] = np.log(log_one_hot_df['number_of_cabin'] + 0.01)



# Feature Elimination
all_df = log_one_hot_df.copy(deep = True)

elimination_features = ['PassengerId', 'Name', 'Cabin', 'Ticket']
for col_name in elimination_features:
    all_df.drop(col_name, axis = 1, inplace = True)

del all_df['Sex-Pclass']
del all_df['Embarked-Pclass']
del all_df['Sex']
del all_df['Embarked']

number_of_train_dataset = len(df_train)

Y_train = all_df.Survived[:number_of_train_dataset]

all_df.drop('Survived', axis = 1, inplace = True)

X_train = all_df[:number_of_train_dataset].values
X_test = all_df[number_of_train_dataset:].values



# Choose Features
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

select = RFE(RandomForestClassifier(n_estimators=100))

select.fit(X_train, Y_train)

X_train_selected = select.transform(X_train)


# Data Analysis
clf = RandomForestClassifier(n_estimators = 100, max_depth = 20, random_state = 0)
# Case1. use 48 features
"""
clf.fit(X_train, Y_train)

idx = (all_df[number_of_train_dataset:].index + 1).tolist()
Y_pre = clf.predict(X_test)
"""
# case2. use 24 features

clf.fit(X_train_selected, Y_train)
idx = (all_df[number_of_train_dataset:].index + 1).tolist()

Y_pre = clf.predict(select.transform(X_test))



submission_columns = ['PassengerId', 'Survived']
submission_df = pd.DataFrame([idx, Y_pre]).T
submission_df.columns = submission_columns
for col_name in submission_columns:
    submission_df[col_name] = submission_df[col_name].astype(int)
submission_df.to_csv("C:/Users/user/Desktop/Machine Learning Practice/Titanic/submission.csv", index = False)




