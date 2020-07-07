import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)

imdb = keras.datasets.imdb
#该数据集已经经过预处理，评论（单词序列）已经被转换为整数序列，其中每个整数表示字典中的特定单词
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

# print("train_data:{}, labels{}".format(len(train_data),len(train_labels)))

# print(train_data[0])
# 如[1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 48
# print(len(train_data[0]), len(train_data[1]))  #218 189
#电影评论可能具有不同的长度。以下代码显示了第一条和第二条评论的中单词数量。由于神经网络的输入必须是统一的长度，我们稍后需要解决这个问题。

# 将整数转换回单词
word_index = imdb.get_word_index()
# print("1111111111111111111111111111")
print(word_index)
word_index = {k:(v+3) for k,v in word_index.items()}
# print("1",word_index)
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3


#value,key值反转  后根据i值来提取value值
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

#用' ' 字符来分割text     for i in text -->  get(i,'?')是根据i 来获取value值
def decode_review(text):
    return ' '.join([reverse_word_index.get(i,'?') for i in text])

print(decode_review(train_data[0])) #<START> this film was just brilliant casting location scenery story direction everyone's really suited
#矩阵
# 将数组转换为表示单词出现与否的由 0 和 1 组成的向量，类似于 one-hot 编码。 需要大量的内存，需要一个大小为 num_words * num_reviews 的矩阵。
#张量
# 或者，我们可以填充数组来保证输入数据具有相同的长度，然后创建一个大小为 max_length * num_reviews 的整型张量。我们可以使用能够处理此形状数据的嵌入层作为网络中的第一层。

# 参数
# padding 字符串，“ pre”或“ post”（可选，默认为“ pre”）：在每个序列之前或之后填充。
# value 指填充的值 此处选择固定0
#为使电影影评长度统一相同 使用pad_sequences来使长度标准化
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                           maxlen=256,
                                           padding="post",
                                           value=word_index['<PAD>'])

#检查现在统一长度之后的值
# print(len(train_data[0]), len(train_data[1]))  #256 256

#查看下首条评论

# print(decode_review(train_data[0]))  在句尾添加了 PAD标识符 shared with us all <PAD> <PAD> <PAD> <PAD>

vocab_size = 10000

model = keras.Sequential()  ######疑问 embedding中 vocab_size的意义  ？？？？ 2020-07-07
model.add(keras.layers.Embedding(vocab_size,16))  #该层是嵌入层 使用了one-hot编码  向量向输出数组增加了一个维度。得到的维度为：(batch, sequence, embedding) 10000输入shape 16输出shape
model.add(keras.layers.GlobalAveragePooling1D())  #GlobalAveragePooling1D 将通过对序列维度求平均值来为每个样本返回一个定长输出向量
model.add(keras.layers.Dense(16,activation="relu"))
model.add(keras.layers.Dense(1,activation="sigmoid"))

model.summary()

#若隐藏层太多 就会导致学习到的更复杂 就会导致过拟合

# 构建损失函数和 优化器
model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])
print(train_data.shape)
#创建一个验证集
x_valid = train_data[:10000]
partial_x_train = train_data[10000:]

y_valid = train_labels[:10000]
partial_y_train = train_labels[10000:]


#训练模型

history = model.fit(partial_x_train,
          partial_y_train,
          batch_size=512,
          epochs=40,
          verbose=1,
          validation_data=(x_valid,y_valid))

#评估模型

results  = model.evaluate(test_data,test_labels,verbose=2)

print(results)
# [0.3270352033615112, 0.8724]

# 创建一个准确率（accuracy）和损失值（loss）随时间变化的图表

#查看history的参数
history_dict = history.history
print(history_dict.keys())  #dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])

# 有四个条目：在训练和验证期间，每个条目对应一个监控指标。我们可以使用这些条目来绘制训练与验证过程的损失值（loss）和准确率（accuracy），以便进行比较

import matplotlib.pyplot as plt
#损失函数的图形绘制
acc = history_dict['acc']
print("acc:",acc)  #acc: [0.63926667, 0.7394, 0.7588, 0.7694,.....]
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
print("len(acc)",len(acc))
epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
# b代表“蓝色实线”
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


#acc准确率的图形绘制

plt.plot(epochs,acc,'bo', label='True_Acc')
plt.plot(epochs,val_acc,'b',label='Val_Acc')
plt.xlabel("epochs")
plt.ylabel('acc')
plt.legend()
plt.show()

#验证过程的损失值（loss）与准确率（accuracy）的情况却并非如此——它们似乎在 20 个 epoch 后达到峰值
#过拟合现象 模型在训练数据上的表现比在以前从未见过的数据上的表现要更好
#在 20 个左右的 epoch 后停止训练来避免过拟合
