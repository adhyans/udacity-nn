import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

data = pd.read_csv('student_data.csv')

def plot_points(data):
    X = np.array(data[['gre', 'gpa']])
    y = np.array(data['admit'])
    admit =  X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], color = 'red')
    plt.scatter([s[0][0] for s in admit], [s[0][1] for s in admit], color = 'cyan')

# Plotting the points
plot_points(data)
# plt.show()

#separating the ranks
data_rank1  = data[data["rank"]==1]
data_rank2  = data[data["rank"]==2]
data_rank3  = data[data["rank"]==3]
data_rank4  = data[data["rank"]==4]

plot_points(data_rank1)
plt.title("Rank 1")
# plt.show()
plot_points(data_rank2)
plt.title("Rank 2")
# plt.show()
plot_points(data_rank3)
plt.title("Rank 3")
# plt.show()
plot_points(data_rank4)
plt.title("Rank 4")
# plt.show()

# Make dummy variables for rank
one_hot_data = pd.concat([data, pd.get_dummies(data['rank'], prefix='rank')], axis=1)

one_hot_data = one_hot_data.drop('rank', axis=1)

processed_data = one_hot_data[:]
processed_data['gre'] = processed_data['gre']/800
processed_data['gpa'] = processed_data['gre']/4.0
processed_data[:10]

sample = np.random.choice(processed_data.index, size=int(len(processed_data) * 0.9), replace=False)
train_data, test_data = processed_data.iloc[sample], processed_data.drop(sample)

print("Number of training samples is", len(train_data))
print("Number of testing samples is", len(test_data))
print(train_data[:10])
print(test_data[:10])

features = np.array(train_data.drop('admit', axis=1))
targets = np.array(keras.utils.to_categorical(train_data['admit'], 2))
features_test = np.array(test_data.drop('admit', axis=1))
targets_test = np.array(keras.utils.to_categorical(test_data['admit'], 2))

print(features[:10])
print(targets[:10])


model = Sequential()
model.add(Dense(128, activation='sigmoid', input_shape=(6,)))
model.add(Dropout(.2))
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(.1))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(features, targets, epochs=200, batch_size=100, verbose=0)

score = model.evaluate(features, targets)
print("\n Training Accuracy:", score[1])
score = model.evaluate(features_test, targets_test)
print("\n Testing Accuracy:", score[1])