import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np

df = pd.read_csv('data/StudentsPerformance.csv')
df.head()

print("\n-----------------------")
print('FULL TABLE: \n')
print(df)
print("-----------------------\n")

# # remove irrelevant data
df.drop(['race/ethnicity', 'lunch', 'parental_level_of_education'], axis='columns', inplace=True)
df.head()

df['average_score'] = df[['math_score', 'reading_score', 'writing_score']].mean(axis=1)

conditions = [
    (df['average_score'] <= 49.49),
    (df['average_score'] >= 49.5)
]

values = [0, 1]

df['PASSED'] = np.select(conditions, values)

print("\n-----------------------")
print('TABLE W/ AVERAGE SCORE & PASSED?: \n')
print(df)
print("-----------------------\n")

inputs = df.drop('PASSED', axis='columns')
target = df.PASSED

dummies_gender = pd.get_dummies(inputs.gender)
dummies_gender.head(3)
dummies_test_prep = pd.get_dummies(inputs.test_preparation_course)
dummies_test_prep.head(3)

inputs = pd.concat([dummies_gender, dummies_test_prep, inputs], axis='columns')
inputs.head

inputs.drop(['gender', 'female', 'test_preparation_course', 'none'], axis='columns', inplace=True)
inputs.head(3)

print('INPUTS WITH DUMMIES')
print(inputs)

# now we use sklearn strain test split method to split our data into training and test sample
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.10)  # train/test samples (90/10)

print("\n-----------------------")
print("SPLIT DATA: \n")
print("X_train: ")
print(len(X_train))
print("Y_train: ")
print(len(y_train))
print("X_test: ")
print(len(X_test))
print("Y_test: ")
print(len(y_test))
print("Inputs: ")
print(len(inputs))
print("-----------------------\n")

print("\n-----------------------")
print("X_train data: \n")
print(X_train)
print("-----------------------\n")
print("\n-----------------------")
print("y_train data: \n")
print(y_train)
print("-----------------------\n")

conditions = [
    (inputs['average_score'] <= 49.49),
    (inputs['average_score'] >= 49.5)
]

values = [0, 1]

inputs['PASSED'] = np.select(conditions, values)

print("\n-----------------------")
print('TABLE W/ AVERAGE SCORE & PASSED?: \n')
print(inputs)
print("-----------------------\n")


# Now we can use NB model
model = GaussianNB()

# we use fit method whenever we want to train the model
model.fit(X_train, y_train)  # when we execute this it will train it

print("\n-----------------------")
print('X_test[:10]:')
print(X_test[:10])

print('\ny_test[:10]:')
print(y_test[:10])

print('\npredict: ')
print(model.predict(X_test[:10]))
print("-----------------------\n")

# after training first thing we can do is measure the score to find the accuracy
print("\n-----------------------")
print("ACCURACY:")
print(model.score(X_test, y_test))
print("-----------------------\n")
