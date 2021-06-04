import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('/home/runner/kaggle/final/dataset.csv')

species=['Iris-setosa','Iris-versicolor','Iris-virginica']

# for flower in species:
#     print(flower)
#     species_df = df[df['Species'] == flower]
#     print('SepalLengthCm')
#     print(sum(species_df['SepalLengthCm'])/len(species_df['SepalLengthCm']))
#     print('SepalWidthCm')
#     print(sum(species_df['SepalWidthCm'])/len(species_df['SepalWidthCm']))
#     print('PetalLengthCm')
#     print(sum(species_df['PetalLengthCm'])/len(species_df['PetalLengthCm']))
#     print('PetalLengthCm')
#     print(sum(species_df['PetalLengthCm'])/len(species_df['PetalWidthCm']))
#     print("\n")
features_to_use=['Species','SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
used_df = df[features_to_use].copy()
for col in used_df.columns:
    if col != 'Species':
        used_df[col] = (used_df[col]-used_df[col].min())/(used_df[col].max() - used_df[col].min())

used_df = used_df.sample(frac=1).reset_index(drop=True)

sample_size=len(np.array(df))//2

df_train = used_df[:sample_size]
df_test = used_df[sample_size:]

train_y = np.array([y for y in np.array(df_train)[:,0]])
train_x = np.array([[y for y in x] for x in np.array(df_train)[:,1:]])
test_y = np.array([y for y in np.array(df_test)[:,0]])
test_x = np.array([[y for y in x] for x in np.array(df_test)[:,1:]])
accuracies=[]
for k in range(1,40):
    KNN = KNeighborsClassifier(n_neighbors = k)
    KNN.fit(train_x,train_y)
    correct = 0
    for i in range(len(test_x)):
        if KNN.predict([test_x[i]]) == test_y[i]:
            correct += 1
    accuracies.append(correct/len(test_x))
print(accuracies)
plt.plot([k for k in range(1,40)], accuracies)
plt.show()
