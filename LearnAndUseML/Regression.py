# 预测房价
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(tf.__version__)

# The Boston Housing Prices dataset 获取数据集
boston_housing = keras.datasets.boston_housing

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# Shuffle the training set 随机洗牌，将数据打乱，但还是对应的
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

# Examples and features 数据集比较小
# 13个特征分别为:
# Per capita crime rate.
# The proportion of residential land zoned for lots over 25,000 square feet.
# The proportion of non-retail business acres per town.
# Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
# Nitric oxides concentration (parts per 10 million).
# The average number of rooms per dwelling.
# The proportion of owner-occupied units built before 1940.
# Weighted distances to five Boston employment centers.
# Index of accessibility to radial highways.
# Full-value property-tax rate per $10,000.
# Pupil-teacher ratio by town.
# 1000 * (Bk - 0.63) ** 2 where Bk is the proportion of Black people by town.
# Percentage lower status of the population.
# 每个特征的数据范围都不尽相同
print("Training set: {}".format(train_data.shape))  # 404 examples, 13 features
print("Testing set:  {}".format(test_data.shape))  # 102 examples, 13 features

print(train_data[0])  # Display sample features, notice the different scales

# 使用dataframe展示数据
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']

df = pd.DataFrame(train_data, columns=column_names)
print(df.head())

# Labels
print(train_labels[0:10])  # Display first 10 entries

# Normalize features 特征标准化

# Test data is not used when calculating the mean and std

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

print(train_data[0])  # First training sample, normalized


# Create the model
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu,
                           input_shape=(train_data.shape[1],)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae'])
    return model


model = build_model()
print(model.summary())


# Train the model
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


EPOCHS = 500

# Store training stats
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[PrintDot()])


def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
             label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
             label='Val loss')
    plt.legend()
    plt.ylim([0, 5])


plot_history(history)

# 在满足条件后自动停止
model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])

plot_history(history)

[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))

# Predict
test_predictions = model.predict(test_data).flatten()
#
# plt.scatter(test_labels, test_predictions)
# plt.xlabel('True Values [1000$]')
# plt.ylabel('Predictions [1000$]')
# plt.axis('equal')
# plt.xlim(plt.xlim())
# plt.ylim(plt.ylim())
# _ = plt.plot([-100, 100], [-100, 100])
#
# error = test_predictions - test_labels
# plt.hist(error, bins = 50)
# plt.xlabel("Prediction Error [1000$]")
# _ = plt.ylabel("Count")

# 总结
# 1.均方误差（MSE）是用于回归问题（不同于分类问题）的一种常用的损失函数。
# 2.类似地，用于回归的评价度量方法与分类不同。一个常见的回归度量方法是平均绝对误差（MAE）。
# 3.当输入数据特征具有不同范围的值时，每个特征应独立地缩放。
# 4.如果没有太多的训练数据，就要选择一个隐藏层较少的小网络来避免过度拟合。
# 5.提前停止是防止过度拟合的有效方法。
