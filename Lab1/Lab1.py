# Dependencies
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

print("***** Train_Set *****")
print(train.head())
print("\n")
print("***** Test_Set *****")
print(test.head())

print("***** Train_Set *****")
print(train.describe())


print(train.columns.values)

# For the train set
train.isna().head()
# For the test set
test.isna().head()

print("*****In the train set*****")
print(train.isna().sum())
print("\n")
print("*****In the test set*****")
print(test.isna().sum())

# # Fill missing values with mean column values in the train set
# train.fillna(train.mean(), inplace=True)
# # Fill missing values with mean column values in the test set
# test.fillna(test.mean(), inplace=True)

numeric_columns = train.select_dtypes(include=[np.number])
train[numeric_columns.columns]=numeric_columns.fillna(numeric_columns.mean())

print("\n")
print("*****Missing values in train*****")
print(train.isna().sum())

numeric_columns = test.select_dtypes(include=[np.number])
test[numeric_columns.columns]=numeric_columns.fillna(numeric_columns.mean())

print("\n")
print("*****Missing values in test*****")
print(test.isna().sum())


train['Ticket'].head()
train['Cabin'].head()

print("\n")
train[['Pclass', 'Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived', ascending=False)
print(train[['Pclass', 'Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived', ascending=False))
print("\n")

train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print("\n")

train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived',ascending=False)
print(train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived',ascending=False))

# g = sns.FacetGrid(train, col='Survived')
# g.map(plt.hist, 'Age', bins=20)
# plt.show()

# grid = sns.FacetGrid(train, col='Survived', row='Pclass', aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend()
# plt.show()

train.info()

#Remove unnessesary features
train = train.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)
test = test.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)

#convert to numerical
labelEncoder = LabelEncoder()
labelEncoder.fit(train['Sex'])
labelEncoder.fit(test['Sex'])
train['Sex'] = labelEncoder.transform(train['Sex'])
test['Sex'] = labelEncoder.transform(test['Sex'])

print("\n")
train.info()
print("\n")
test.info()


X = np.array(train.drop(['Survived'], axis = 1).astype(float))
y = np.array(train['Survived'])
train = train.drop(['Survived'], axis =1)


print("\n")
train.info()
print("\n")

# kmeans = KMeans(n_clusters=2)
# kmeans.fit(X)
# KMeans(algorithm='lloyd', copy_x=True, init='k-means++', max_iter=300, n_clusters=2, n_init=10, random_state=None, tol=0.0001, verbose=0)

# correct = 0
# for i in range(len(X)):
#  predict_me = np.array(X[i].astype(float))
#  predict_me = predict_me.reshape(-1, len(predict_me))
#  prediction = kmeans.predict(predict_me)
#  if prediction[0] == y[i]:
#     correct += 1
# print(correct/len(X))

# kmeans = kmeans = KMeans(n_clusters=2, max_iter=600, algorithm = 'lloyd')
# kmeans.fit(X)
# KMeans(algorithm='lloyd', copy_x=True, init='k-means++', max_iter=600,n_clusters=2, n_init=10, random_state=None, tol=0.0001, verbose=0)

# correct = 0
# for i in range(len(X)):
#  predict_me = np.array(X[i].astype(float))
#  predict_me = predict_me.reshape(-1, len(predict_me))
#  prediction = kmeans.predict(predict_me)
#  if prediction[0] == y[i]:
#     correct += 1
# print(correct/len(X))


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=2, max_iter=600, algorithm = 'lloyd')
kmeans.fit(X_scaled)
KMeans(algorithm='lloyd', copy_x=True, init='k-means++', max_iter=600, n_clusters=2, n_init=10, random_state=None, tol=0.0001, verbose=0)
correct = 0
for i in range(len(X)):
 predict_me = np.array(X[i].astype(float))
 predict_me = predict_me.reshape(-1, len(predict_me))
 prediction = kmeans.predict(predict_me)
 if prediction[0] == y[i]:
    correct += 1
print(correct/len(X))



