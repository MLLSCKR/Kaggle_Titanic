# Titanic Machine Learning Project

# import modules
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import seaborn as sns
import matplotlib.pyplot as plot

# read data from csv files
df_test = pd.read_csv("C:/Users/user/Desktop/Machine Learning Practice/Titanic/test.csv")
df_train = pd.read_csv("C:/Users/user/Desktop/Machine Learning Practice/Titanic/train.csv")

df = pd.concat([df_test, df_train], ignore_index = True)

# 결측치 처리
# 평균값, 최빈값, 중위값 중 평균값을 사용하기로 결정하였다.
#
# 실험 1
# 성별에 따른 나이의 평균값 df.groupby('Sex').Age.mean()
#                         df[df['Age'].notnull()].groupby(['Sex'])['Age'].mean()
# 평균 -> 여성 : 28.687088, 남성 : 30.585228
# 편차 -> 여성 : 14.576962, 남성 : 14.280581
# Sex와 Age 사이의 correlation : 0.063645
#
# 실험 2
# Pclass에 따른 나이의 평균값
# 평균 -> 1등급 : 39.159930, 2등급 : 29.506705, 3등급 : 24.816367
# 편차 -> 1등급 : 14.548028, 2등급 : 13.638627, 3등급 : 11.958202
# Pclass와 Age 사이의 correlation : -0.408106
#
# Age와의 Correlation이 가장 높은 것은 Pclass이기에 Pclass에 따른 평균으로 결측치를 보정한다.

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

# to check data figure
"""
temp_columns = ['Sex', 'Pclass', 'Embarked']

for col_name in temp_columns:
    temp_df = pd.merge(one_hot_df[col_name], df['Survived'], left_index = True, right_index = True)
    sns.countplot(x = "Survived", hue = col_name, data = temp_df)
    plot.show()
"""
"""
temp_columns = ['Sex', 'Pclass', 'Embarked']

temp_df = pd.merge(one_hot_df[temp_columns], one_hot_df['Survived'], left_index = True, right_index = True)
g = sns.catplot(x = "Embarked", hue = "Pclass", col = "Survived", data = temp_df, kind = "count")
"""

# to check corr
"""
crosscheck_columns = one_hot_df.columns[-8:]
temp_df = pd.merge(one_hot_df[crosscheck_columns], one_hot_df['Survived'], left_index=True, right_index=True)

corr_result = temp_df.corr()
"""
# to check corr in heat map
# sns.heatmap(corr_result, annot = True)


# Feature Scaling
def feature_scaling(df, scaling_strategy="min-max", column = None):
    if column == None:
        column = [column_name for column_name in df.columns]
    for column_name in column:
        if scaling_strategy == "min-max":
            df[column_name] = (df[column_name] - df[column_name].min()) / (df[column_name].max() - df[column_name].min())
        elif scaling_strategy == "z-score":
            df[column_name] = (df[column_name] - df[column_name].mane()) / (df[column_name].std())
    return df

# Feature Engineering
# Generation
#     Binarization, Quantization, Scaling, Interaction features, Log Transformation ...
# Selection
#     Univariate station, ...


# Log transformation
# 데이터의 분포가 극단적으로 모였을 때
# 선형 모델은 데이터가 정규분포때 적합하다.
# 기존의 경우
"""
fig = plot.figure()
fig.set_size_inches(10, 5)

ax = []
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
for i, col_name in enumerate(numeric_columns):
    ax.append(fig.add_subplot(2, 2, i+1))
    X_1 = one_hot_df[col_name]
    
    ax[i] = sns.distplot(X_1, bins = 10)
    ax[i].set_title(col_name)
"""
# log의 경우
"""
fig = plot.figure()
fig.set_size_inches(10, 5)

ax = []
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
for i, col_name in enumerate(numeric_columns):
    ax.append(fig.add_subplot(2, 2, i+1))
    X_1 = np.log10(one_hot_df[col_name]+0.5)
    
    ax[i] = sns.distplot(X_1, bins = 10)
    ax[i].set_title(col_name)
"""


# Mean Encoding
# Mean Encoding 사용시 corr이 더 증가하는 것을 확인할 수 있음
"""
temp_df = pd.merge(one_hot_df['Pclass'], one_hot_df['Survived'], left_index = True, right_index = True)
temp_df['Pclass'].replace((temp_df.groupby('Pclass')['Survived'].mean()))
"""

temp_columns = ['Sex', 'Pclass', 'Embarked']

me_list = []

for col_name in temp_columns:
    temp_df = pd.merge(one_hot_df[col_name], df['Survived'], left_index = True, right_index = True)
    temp_df["me_"+col_name] = temp_df[col_name].replace(temp_df.groupby(col_name)['Survived'].mean())
    #sns.countplot(x = col_name, hue="Survived", data = temp_df)
    #plot.show()
    
    #sns.countplot(x = "me_"+col_name, hue="Survived", data = temp_df)
    #plot.show()
    
    me_list.append(temp_df.drop("Survived", axis = 1))
    
temp_df = pd.merge(pd.concat(me_list, axis=1)[["me_" + col_name for col_name in temp_columns]], one_hot_df['Survived'], left_index = True, right_index = True)


# Categorical Combination

one_hot_df['Sex-Pclass'] = df['Sex'].map(str) + df['Pclass'].map(str)
one_hot_df['Embarked-Pclass'] = df['Embarked'].map(str) + df['Pclass'].map(str)

one_hot_df = merge_and_get(
    one_hot_df, pd.get_dummies(one_hot_df['Sex-Pclass'], prefix = 'SexPclass'), on = None, index = True)
one_hot_df = merge_and_get(
    one_hot_df, pd.get_dummies(one_hot_df['Embarked-Pclass'], prefix = 'EmbarkedPclass'), on = None, index = True)

crosscheck_columns = one_hot_df.columns[-15:]
temp_df = pd.merge(one_hot_df[crosscheck_columns], one_hot_df['Survived'], left_index=True, right_index=True)

corr_result = temp_df.corr()

# Data Visualization
"""
temp_df = pd.merge(one_hot_df[numeric_columns], one_hot_df['Survived'], left_index = True, right_index = True)
sns.pairplot(temp_df)

corr = temp_df.corr()
sns.set()
plot.subplots(figsize=(20,15))
ax = sns.heatmap(corr, annot = True, linewidths = 8)

sns.barplot(x = "SibSp", y = "Fare", hue = "Survived", data = temp_df, ci = 68, capsize = .2)
"""



