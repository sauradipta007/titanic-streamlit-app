import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head()

# Fix the pandas warnings by using proper assignment
train.loc[:, 'Age'] = train['Age'].fillna(train['Age'].median())
train.loc[:, 'Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

train.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)

# Convert categorical features to numerical BEFORE splitting data
train.loc[:, 'Sex'] = train['Sex'].map({'male': 0, 'female': 1})
train.loc[:, 'Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

print("\nStatistical Summary of the Dataset:")
print(train.describe())

print("\nSurvival Rate Distribution:")
print(train['Survived'].value_counts(normalize=True))

print("\nChecking data before plotting:")
print("Number of rows in dataset:", len(train))
print("Unique values in Sex column:", train['Sex'].unique())
print("Unique values in Survived column:", train['Survived'].unique())

# Set a different style that's available by default
plt.style.use('default')
plt.figure(figsize=(8, 6))

# Create the plot
sns.countplot(x='Survived', hue='Sex', data=train)
plt.title("Survival by Sex")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

#logistic regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = train.drop('Survived', axis=1)
y = train['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))

#test the model
test['Age'].fillna(train['Age'].median(), inplace=True)
test['Fare'].fillna(train['Fare'].median(), inplace=True)
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})
test['Embarked'] = test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
test.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)

predictions = model.predict(test.drop('PassengerId', axis=1))

submission = pd.DataFrame({
    "PassengerId": test['PassengerId'],
    "Survived": predictions
})
submission.to_csv('submission.csv', index=False)

