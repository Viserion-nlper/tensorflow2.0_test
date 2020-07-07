# 使用评论文本将影得分为积极（正）或消极（nagetive）两类
# 使用Tensorflow Hub和Keras进行迁移学习的基本应用

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import tensorflow as tf
tf.enable_eager_execution()  #
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)


train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
# print(train_examples_batch)
#
# print(train_labels_batch)
# 建立模型
# 神经网络由堆叠的层构建
#如何表示文本？
#模型里有多少层？
#每个层中有多少个隐藏单元？
#文本表示 embeddings vectors   这里可以用预先处理好的text embedding作为首层 这样有1.不用担心文本的预处理 2.可以从迁移学习中受益 3.嵌入会有固定长度 更加方便处理
#这里使用TensorFlow Hub 中名为 google/tf2-preview/gnews-swivel-20dim/1(https://hub.tensorflow.google.cn/google/tf2-preview/gnews-swivel-20dim/1?hl=zh_cn) 的预训练text embedding
# 1. google/tf2-preview/gnews-swivel-20dim-with-oov/1  (https://hub.tensorflow.google.cn/google/tf2-preview/gnews-swivel-20dim-with-oov/1?hl=zh_cn)
# ——类似 google/tf2-preview/gnews-swivel-20dim/1，但 2.5%的词汇转换为未登录词桶（OOV buckets）。如果任务的词汇与模型的词汇没有完全重叠，这将会有所帮助。
# 2. google/tf2-preview/nnlm-en-dim50/1  (https://hub.tensorflow.google.cn/google/tf2-preview/nnlm-en-dim50/1?hl=zh_cn)
# ——一个拥有约 1M 词汇量且维度为 50 的更大的模型。
# 3. google/tf2-preview/nnlm-en-dim128/1 ——拥有约 1M 词汇量且维度为128的更大的模型(https://hub.tensorflow.google.cn/google/tf2-preview/nnlm-en-dim128/1?hl=zh_cn)
# ——拥有约 1M 词汇量且维度为128的更大的模型。


#使用TensorflowHub进行词嵌入
embedding = "https://hub.tensorflow.google.cn/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding,input_shape=[],trainable=True,dtype=tf.string)
print(hub_layer(train_examples_batch[:3]))

#构建完整模型
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16,activation="relu",name="dense16"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid",name="dense1")) # Sigmoid 激活函数，其函数值为介于 0 与 1 之间的浮点数，表示概率或置信水平。
model.summary()
#损失函数与优化器
#由于是二分类问题模型输出为概率值 sigmoid函数激活  使用binary_crossentropy 损失函数  还可以选择mean_squared_error
# 一般来说 binary_crossentropy 更适合处理概率——它能够度量概率分布之间的“距离”  度量真实值和预测值之间的距离
# 在研究回归问题时  可以选择使用均方误差MSE来衡量

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

history = model.fit(train_data.shuffle(10000).batch(512),
          epochs=5,
          validation_data=validation_data.batch(512),
          verbose=1)

#评估模型
results = model.evaluate(test_data.batch(512),verbose=2)
for name,value in zip(model.metrics_names, results):
    print('%s: %.3f'% (name, value))
# loss: 0.500
# acc: 0.762