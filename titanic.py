import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('data/titanic.csv')
df.head()

print("\n-----------------------")
print('FULL TABLE: \n')
print(df)
print("-----------------------\n")

# remove irrelevant data
df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns', inplace=True)
df.head()

print("\n-----------------------")
print('TABLE WITH REMOVED DATA: \n')
print(df)
print("-----------------------\n")

inputs = df.drop('Survived', axis='columns')
target = df.Survived

print("\n-----------------------")
print('TABLE WITH TARGET DATA (SURVIVED): \n')
print(target)
print("-----------------------\n")

# inputs.Sex = inputs.Sex.map({'male': 1, 'female': 2})
# convert sex column into 0s and 1s
dummies = pd.get_dummies(inputs.Sex)
dummies.head(3)
# dummies will have 2 different columns and put 0s and 1s values

# we're append these dummies columns to our input dataframe and we do it by using pandas concat
# in the result now we have female and male columns
inputs = pd.concat([inputs, dummies], axis='columns')
inputs.head(3)

# dropping female column because we don't need both female and male
inputs.drop(['Sex', 'female'], axis='columns', inplace=True)
inputs.head(3)

# we also want to find out if there are any numbers in any columns
inputs.columns[inputs.isna().any()]  # will tell us that age column has any values

inputs.Age[:10]

# if we get NaN we can use mean which fills in the NaN values
inputs.Age = inputs.Age.fillna(inputs.Age.mean())
inputs.head(5)

print("\n-----------------------")
print("INPUTS: \n", inputs)
print("-----------------------\n")

# now we use sklearn strain test split method to split our data into training and test sample
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2)  # train/test samples (80/20)

print("\n-----------------------")
print("TRAIN DATA: \n")
print("X_train: ")
print(len(X_train))
print("X_test: ")
print(len(X_test))
print("Inputs: ")
print(len(inputs))
print("-----------------------\n")

print("\n-----------------------")
print("X_train data: \n")
print(X_train)
print("-----------------------\n")

model = GaussianNB()

# we use fit method whenever we want to train the model
model.fit(X_train, y_train)  # when we execute this it will train it

# after training first thing we can do is measure the score to find the accuracy
print("\n-----------------------")
print("ACCURACY:")
print(model.score(X_test, y_test))
print("-----------------------\n")

print("\n-----------------------")
print("X_test[0:10]: \n")
print(X_test[0:10])
print("-----------------------\n")
print("\n-----------------------")
print("y_test[0:10]: \n")
print(y_test[0:10])

print("\n\nTEST AND SEE IF PREDICT IS EQUAL TO y_test:")
# we can test and see if predict is equal to y_test
print(model.predict(X_test[0:10]))
print("-----------------------\n")

# we can also use predict_proba function to figure out what are probabilities of each class (survived or not)
print("\n-----------------------")
print(model.predict_proba(X_test[:10]))
print("-----------------------\n")
