import pandas as pd
import numpy as np

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

# Parch
df['Parch>0'] = df['Parch'].apply(indicator_greater_than_zero)

# cabinType

df['Cabin']= df['Cabin'].fillna('None')

def get_cabin_type(cabin):
    if cabin != 'None':
        return cabin[0]
    else:
        return cabin

df['CabinType'] = df['Cabin'].apply(get_cabin_type)

for cabin_type in df['CabinType'].unique():
    dummy_variable_name = 'CabinType={}'.format(cabin_type)
    dummy_variable_values = df['CabinType'].apply(lambda entry: int(entry==cabin_type))
    df[dummy_variable_name] = dummy_variable_values

del df['CabinType']

# embarked

df['Embarked'] = df['Embarked'].fillna('None')

for cabin_type in df['Embarked'].unique():
    dummy_variable_name = 'Embarked={}'.format(cabin_type)
    dummy_variable_values = df['Embarked'].apply(lambda entry: int(entry==cabin_type))
    df[dummy_variable_name] = dummy_variable_values

del df['Embarked']


print(df)