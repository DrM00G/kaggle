import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression, LogisticRegression

df = pd.read_csv('/home/runner/kaggle/processed_data.csv')

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

ratings = df['Survived']


best_features=[x for x in df.columns if x!="id" and x!='Survived']


    # print(best_features+[feature])
test_df=df.copy()[best_features]
selection_ratings=ratings.copy()

num_train = 500

y_train = np.array(selection_ratings[:500])
y_test = np.array(selection_ratings[500:])

X_train = np.array(test_df[:500])
X_test = np.array(test_df[500:])

regressor = LogisticRegression(max_iter=1000)
regressor.fit(X_train, y_train)
cols = ['Constant']+[x for x in test_df.columns if x != 'Survived']
coefs = [regressor.intercept_[0]]+[x for x in regressor.coef_[0]]

train_test=[]

for data in [(X_train,y_train),(X_test,y_test)]:
    predictions = regressor.predict(data[0])

    result = [0,0]
    for i in range(len(predictions)):
        output = 1 if predictions[i]>0.5 else 0
        result[1]+=1
        if output == data[1][i]:
            result[0]+=1

    train_test.append(result[0]/result[1])


acc=train_test[1]   
for feature in [x for x in best_features]:
    # print([x for x in best_features if x!=feature])
    test_df=df.copy()[[x for x in best_features if x!=feature]]
    selection_ratings=ratings.copy()

    num_train = 500

    y_train = np.array(selection_ratings[:500])
    y_test = np.array(selection_ratings[500:])

    X_train = np.array(test_df[:500])
    X_test = np.array(test_df[500:])

    regressor = LogisticRegression(max_iter=1000)
    regressor.fit(X_train, y_train)
    cols = ['Constant']+[x for x in test_df.columns if x != 'Survived']
    coefs = [regressor.intercept_[0]]+[x for x in regressor.coef_[0]]

    train_test=[]

    for data in [(X_train,y_train),(X_test,y_test)]:
        predictions = regressor.predict(data[0])

        result = [0,0]
        for i in range(len(predictions)):
            output = 1 if predictions[i]>0.5 else 0
            result[1]+=1
            if output == data[1][i]:
                result[0]+=1

        train_test.append(result[0]/result[1])
    if train_test[1]>acc:
        best_features.remove(feature)

print(best_features)
print(acc)