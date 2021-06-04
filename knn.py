import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('/home/runner/kaggle/processed_data.csv')

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


features_to_use = ['Survived', "Sex", "Pclass", "Fare", "Age", "SibSp"]

k_set = [1,3,5,10,15,20,30,40,50,75]
def leave_one_out_validation(x,y, KNN, pred_col,iterations):
    correct = 0
    for i in range(iterations):
        removed_row = x[i]
        removed_result = y[i]
        leave_one_out_y = np.delete(y,i, axis = 0)
        leave_one_out_x = np.delete(x,i,axis = 0)
        
        if KNN.fit(leave_one_out_x, leave_one_out_y).predict([removed_row]) == removed_result:
            correct+=1
        
    return correct/len(df.index)

# train_y = np.array([y for y in np.array(df_train)[:,0]])
# train_x = np.array([[y for y in x] for x in np.array(df_train)[:,1:]])
# test_y = np.array([y for y in np.array(df_test)[:,0]])
# test_x = np.array([[y for y in x] for x in np.array(df_test)[:,1:]])

data_y = np.array([y for y in np.array(df)[:,0]])
data_x = np.array([[y for y in x] for x in np.array(df)[:,1:]])

prediction_accuracies = []
for k in k_set:
    knn = KNeighborsClassifier(n_neighbors = k)
    prediction_accuracies.append(leave_one_out_validation(df,k))

print(prediction_accuracies)
plt.plot(k_set, prediction_accuracies)

plt.savefig('Titanic KNN accuracy.png')
