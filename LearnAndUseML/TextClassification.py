# 电影评论分类
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


print(tf.__version__)

# Download the IMDB dataset
imdb = keras.datasets.imdb
# 训练集，测试集
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Explore the data 数据全部是数字
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0])
# 不同评论的长度可能不同
len(train_data[0]), len(train_data[1])

# Convert the integers back to words 可以使用配套字典将数字转化为评论文字
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


print(decode_review(train_data[0]))
print(decode_review(train_data[1]))

# Prepare the data
# 必须把数组转化成tensor，有两种方法
# 方法一：将数组转化成0,1向量，比如[3,5]转化成一万维的向量，除了索引3,5处为1，其余为0，但是这种方法是内存密集型的，需要num_words * num_reviews的矩阵
# 方法二：填充数组，使它们都有相同长度，然后创建一个大小为max_length * num_reviews的整形张量
# 在这个教程中，使用第二个方法，首先处理数据，标准化长度
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
# 此时长度均为256，结尾原来无数据处填0
print(len(train_data[0]), len(train_data[1]))
print(train_data[0])

# Build the model
# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
# The first layer is an Embedding layer. This layer takes the integer-encoded vocabulary and looks up the embedding vector for each word-index.
# These vectors are learned as the model trains. The vectors add a dimension to the output array. The resulting dimensions are: (batch, sequence, embedding).
model.add(keras.layers.Embedding(vocab_size, 16))
# Next, a GlobalAveragePooling1D layer returns a fixed-length output vector for each example by averaging over the sequence dimension.
# This allows the model can handle input of variable length, in the simplest way possible.
model.add(keras.layers.GlobalAveragePooling1D())
# This fixed-length output vector is piped through a fully-connected (Dense) layer with 16 hidden units.
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
# The last layer is densely connected with a single output node. Using the sigmoid activation function, this value is a float between 0 and 1, representing a probability, or confidence level.
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

print(model.summary())

# Hidden units
# 隐藏层，更多的隐藏层可以进行更复杂的学习

# Loss function and optimizer
# loss——损失函数
# metrics——监测训练集和测试集
# optimizer——决定数据如何更新
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Create a validation set 验证集，从训练集中划分出来
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# Train the model
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# Evaluate the model
results = model.evaluate(test_data, test_labels)

print(results)

# Create a graph of accuracy and loss over time
history_dict = history.history
print(history_dict.keys())

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()