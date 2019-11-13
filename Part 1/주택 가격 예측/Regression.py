from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

#데이터 정규화
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

# 샘플이 적으므로 64 * 2 의 작은 은닉층을 구성

from keras import models, layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1], )))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# K-fold cross validation
import numpy as np

k = 4

num_val_samples = len(train_data) // k      #몫
num_epochs = 500
all_mae_histories = []

for i in range(k):
    print('처리중인 폴드 #',i)
    val_data = train_data[i * num_val_samples : (i+1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples : (i+1) * num_val_samples]

    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples], train_data[(i+1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples], train_targets[(i+1) * num_val_samples:]],
        axis=0)

    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    #verbose=0이면 훈련 과정을 출력하지 않음
    mae_history = history.history['mae']
    all_mae_histories.append(mae_history)

avg_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

def smooth_curve(points, factor=0.9):
    smoothed_point = []
    for point in points:
        if smoothed_point:
            previous = smoothed_point[-1]
            smoothed_point.append(previous * factor + point * (1-factor))
        else:
            smoothed_point.append(point)
    return smoothed_point

import matplotlib.pyplot as plt

smoothed_mae_history = smooth_curve(avg_mae_history[10:])

plt.plot(range(1, len(smoothed_mae_history)+1), smoothed_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

model = build_model()
model.fit(train_data, train_targets, epochs=80, batch_size=16)
mse_score, mae_score = model.evaluate(test_data, test_targets)

print(mse_score)
print(mae_score)