# linear algebra
import numpy as np

# data processing
import pandas as pd

# data visualization
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

test_df = pd.read_csv("test.csv")
train_df = pd.read_csv("train.csv")

# # summary
# train_df.info()
# print()
#
# # count, mean, std...
# print(train_df.describe())
# print()
#
# # first eight lines
# print(train_df.head(10))
# print()
#
# # top 5 missing data
# total = train_df.isnull().sum().sort_values(ascending=False)
# percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100
# percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
# missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
# print(missing_data.head(10))
# print()
#
# # see headers
# print(train_df.columns.values)
# print()

# # Men and Women survive plot
# # Assuming 'train_df' has 'Sex', 'Age', and 'Survived' columns
# survived = 'survived'
# not_survived = 'not survived'
#
# # Create subplots for the gender-based distribution
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
# # Filter data for females and males
# women = train_df[train_df['Sex'] == 'female']
# men = train_df[train_df['Sex'] == 'male']
# # Plot for females (survived vs. not survived)
# sns.histplot(women[women['Survived'] == 1]['Age'].dropna(), bins=18, label=survived, ax=axes[0], kde=False, color='blue')
# sns.histplot(women[women['Survived'] == 0]['Age'].dropna(), bins=40, label=not_survived, ax=axes[0], kde=False, color='red')
# axes[0].legend()
# axes[0].set_title('Female')
# # Plot for males (survived vs. not survived)
# sns.histplot(men[men['Survived'] == 1]['Age'].dropna(), bins=18, label=survived, ax=axes[1], kde=False, color='blue')
# sns.histplot(men[men['Survived'] == 0]['Age'].dropna(), bins=40, label=not_survived, ax=axes[1], kde=False, color='red')
# axes[1].legend()
# axes[1].set_title('Male')
# plt.tight_layout()
# plt.show()
#
# # Create a FacetGrid to separate by 'Embarked'
# FacetGrid = sns.FacetGrid(train_df, row='Embarked', height=4.5, aspect=1.6)
# FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', hue='Sex', data=train_df, palette=None,  order=None, hue_order=None)
# FacetGrid.add_legend()
# plt.show()
#
# # Pclass plot
# sns.barplot(x='Pclass', y='Survived', data=train_df)
# plt.show()
#
# # Create a FacetGrid based on 'Survived' and 'Pclass'
# grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend()
# plt.show()

data = [train_df, test_df]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)
# print(train_df['not_alone'].value_counts())
# print()
#
# # SibSp and Parch
# sns.catplot(x='relatives', y='Survived', data=train_df, kind='point', aspect=2.5)
# plt.show()

# PassengerId: drop irrelevant attribute
train_df = train_df.drop(['PassengerId'], axis=1)

# Cabin: we convert it to deck info, then drop cabin
import re
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
data = [train_df, test_df]

for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)
train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)

data = [train_df, test_df]

# Age: fill in the missing values with random numbers that computed based on the mean age value in regards to the standard deviation and is_null
for dataset in data:
    mean = train_df["Age"].mean()
    std = test_df["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_df["Age"].astype(int)
train_df["Age"].isnull().sum()

# Embarked: fill 2 missing values with the most common value, which is S
# print(train_df['Embarked'].describe())
common_value = 'S'
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)

# Fare: Converting “Fare” from float to int64
data = [train_df, test_df]

for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype('float')
    dataset['Fare'] = dataset['Fare'].astype(int)

# Name: extract the Titles from the Name, so that we can build a new feature out of that.
data = [train_df, test_df]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    # extract titles
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\\.', expand=False)
    # replace titles with a more common title or as Rare
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # convert titles into numbers
    dataset['Title'] = dataset['Title'].map(titles)
    # filling NaN with 0, to get safe
    dataset['Title'] = dataset['Title'].fillna(0)
train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)

# Sex: convert to numeric
genders = {"male": 0, "female": 1}
data = [train_df, test_df]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)

# Ticket: convert to numeric
train_df = train_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1)

# Embarked: fill 2 missing values with the most common value, which is S
ports = {"S": 0, "C": 1, "Q": 2}
data = [train_df, test_df]
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)

# Age: categorize
data = [train_df, test_df]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6
# print(train_df['Age'].value_counts())

# Fare: use sklearn “qcut()” to categorize. pd.qcut() categorizes the data based on quantiles
data = [train_df, test_df]
for dataset in data:
    dataset['Fare'] = pd.qcut(dataset['Fare'], 6, labels=False)  # 6 categories, labels=False gives bin numbers 0-5
# print(train_df['Fare'].value_counts())

# Age times Class
data = [train_df, test_df]
for dataset in data:
    dataset['Age_Class']= dataset['Age']* dataset['Pclass']

# Fare per Person
for dataset in data:
    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)
    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()

# Stochastic Gradient Descent (SGD)
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
sgd.score(X_train, Y_train)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
print(f"Accuracy of SGD Classifier on training set: {acc_sgd}%")

# Random Forest:
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(f"Accuracy of Random Forest on training set: {acc_random_forest}%")

# Logistic Regression:
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print(f"Accuracy of Logistic Regression on training set: {acc_log}%")

# K Nearest Neighbor:
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print(f"Accuracy of KNN on training set: {acc_knn}%")

# Gaussian Naive Bayes:
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
print(f"Accuracy of Gaussian Naive Bayes on training set: {acc_gaussian}%")

# Perceptron:
perceptron = Perceptron(max_iter=500)
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print(f"Accuracy of Perceptron on training set: {acc_perceptron}%")

# Linear Support Vector Machine:
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print(f"Accuracy of Linear Support Vector Machine on training set: {acc_linear_svc}%")

# Decision Tree:
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print(f"Accuracy of Decision Tree on training set: {acc_decision_tree}%")
print()

results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Stochastic Gradient Decent',
              'Decision Tree'],
    'Score': [acc_linear_svc, acc_knn, acc_log,
              acc_random_forest, acc_gaussian, acc_perceptron,
              acc_sgd, acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
print(result_df.head(9))
print()

rf = RandomForestClassifier(n_estimators=100, oob_score=True)
rf.fit(X_train, Y_train)
scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
print()

# Analyze the importance of each feature in a Random Forest model
# importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})
# importances = importances.sort_values('importance',ascending=False).set_index('feature')
# print(importances.head(15))
# importances.plot.bar()
# plt.show()
# So Parch and not_alone are not really import, can choose to drop, in case of overfitting

print("oob score:", round(rf.oob_score_, 4)*100, "%")
