import pandas 
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import sys

df = pd.read_csv('/home/runner/kaggle/dataset.csv')

keep_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
df = df[keep_cols]

# process the columns

# sex
def convert_sex_to_int(sex):
    if sex == 'male':
        return 0
    elif sex == 'female':
        return 1

df['Sex'] = df['Sex'].apply(convert_sex_to_int)

#age
age_nan = df['Age'].apply(lambda entry: np.isnan(entry))
age_not_nan = df['Age'].apply(lambda entry: not np.isnan(entry))

mean_age = df['Age'][age_not_nan].mean()

df.loc[age_nan, ['Age']] = mean_age

#sibsp
def indicator_greater_than_zero(x):
    if x > 0:
        return 1
    else:
        return 0

df['SibSp>0'] = df['SibSp'].apply(indicator_greater_than_zero)

# parch
df['Parch>0'] = df['Parch'].apply(indicator_greater_than_zero)

print(df)
# CabinType

# df['Cabin']= df['Cabin'].fillna('None')

# def get_cabin_type(cabin):
#     if cabin != 'None':
#         return cabin[0]
#     else:
#         return cabin

# df['CabinType'] = df['Cabin'].apply(get_cabin_type)

# for cabin_type in df['CabinType'].unique():
#     dummy_variable_name = 'CabinType={}'.format(cabin_type)
#     dummy_variable_values = df['CabinType'].apply(lambda entry: int(entry==cabin_type))
#     df[dummy_variable_name] = dummy_variable_values

# del df['CabinType']

# # Embarked

# df['Embarked'] = df['Embarked'].fillna('None')

# for cabin_type in df['Embarked'].unique():
#     dummy_variable_name = 'Embarked={}'.format(cabin_type)
#     dummy_variable_values = df['Embarked'].apply(lambda entry: int(entry==cabin_type))
#     df[dummy_variable_name] = dummy_variable_values

# del df['Embarked']

# #print(df)


# #print(df['CabinType'])


# features_to_use = ['Sex', 'Pclass', 'Fare', 'Age', 'SibSp', 'SibSp>0', 'Parch>0', 'Embarked=C', 'Embarked=None', 'Embarked=Q', 'Embarked=S', 'CabinType=A', 'CabinType=B', 'CabinType=C', 'CabinType=D', 'CabinType=E', 'CabinType=F', 'CabinType=G', 'CabinType=None', 'CabinType=T']

# #print(df.columns)

# # split into training/testing dataframes
# num_train = 500
# df_train = df[:num_train]
# df_test = df[num_train:]

# arr_train = np.array(df_train[df_train])
# arr_test = np.array(df_test[df_test])

# y_train = arr_train[:,0]
# y_test = arr_test[:,0]

# X_train = arr_train[:,1:]
# X_test = arr_test[:,1:]

# regressor = LinearRegression()
# regressor.fit(X_train, y_train)

# coef_dict = {}
# feature_columns = df_train.columns[1:]
# feature_coefficients = regressor.coef_[0]
# #print('feature_coefficients', feature_coefficients)
# #print(feature_coefficients)
# for i in range(len(feature_columns)):
#         column = feature_columns[i]
#         coefficient = feature_coefficients[i]
#         coef_dict[column] = coefficient

# y_test_predictions = regressor.predict(X_test)
# print(y_test_predictions)
# y_train_predictions = regressor.predict(X_train)

# def convert_regressor_output_to_survival_value(output):
#     if output < 0.5:
#         return 0
#     else:
#         return 1

# y_test_predictions = [convert_regressor_output_to_survival_value(output) for output in y_test_predictions]
# y_train_predictions = [convert_regressor_output_to_survival_value(output) for output in y_train_predictions]

# def get_accuracy(predictions, actual):
#     num_correct = 0
#     num_incorrect = 0
#     for i in range(len(predictions)):
#         if predictions[i] == actual[i]:
#             num_correct += 1
#         else:
#             num_incorrect += 1
    
#     return num_correct / (num_correct + num_incorrect)

# training_accuracy = get_accuracy(y_train_predictions, y_train)
# testing_accuracy = get_accuracy(y_test_predictions, y_test)

