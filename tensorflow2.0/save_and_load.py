# !pip install -q pyyaml h5py  # 需要以 HDF5 格式保存模型


from __future__ import absolute_import, division, print_function, unicode_literals

import os

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data
print(tf.version.VERSION)

mnist = input_data.read_data_sets('G:\尚学堂tensorflow\mnist数据集')

train_X = mnist.train.images
validation_X = mnist.validation.images
test_X = mnist.test.images

train_Y = mnist.train.labels
validation_Y = mnist.validation.labels
test_Y = mnist.test.labels
# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
#
train_labels = train_Y[:10000]
test_labels  = test_Y[:10000]

train_images = train_X[:10000].reshape(-1,28 *28) /255.0
test_images  = test_X[:10000].reshape(-1, 28 * 28) / 255.0


# print(train_images[1])
#构建简单模型

# 定义一个简单的序列模型
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  return model

# 创建一个基本的模型实例
model = create_model()

# 显示模型的结构
model.summary()


#在训练期间保存模型（以 checkpoints 形式保存）

# 使用训练好的模型而无需从头开始重新训练，或在打断的地方开始训练，以防止训练过程没有保存。
# tf.keras.callbacks.ModelCheckpoint 允许在训练的过程中和结束时回调保存的模型
# 创建一个只在训练期间保存权重的 tf.keras.callbacks.ModelCheckpoint 回调
# checkpoint_path = "./model/training_1/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
#
# # 创建一个保存模型权重的回调
# cp_callback  = keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,save_weights_only=True,
#                                 verbose=1)
#
# # print(cp_callback)
#
# 使用一个新的回调模型训练
# history = model.fit(train_images,train_labels,batch_size=256,
#           epochs=10,
#           verbose=2,
#           callbacks=[cp_callback], # 通过回调训练
#           validation_data=(test_images,test_labels))
#
# print(history.history.keys())
# 创建一个新的未经训练的模型。仅恢复模型的权重时，
# 必须具有与原始模型具有相同网络结构的模型。由于模型具有相同的结构
# ，您可以共享权重，尽管它是模型的不同实例。 现在重建一个新的未经训练的模型
# ，并在测试集上进行评估。未经训练的模型将在机会水平（chance levels）上执行（准确度约为10％）：

#未经过训练的model
# 创建一个基本模型实例
# model = create_model()

# 评估模型
# loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
# print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
#
# # 然后从 checkpoint 加载权重并重新评估：
# model.load_weights(checkpoint_dir)   #使用checkpoint_dir文件进行加载model文件
# loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# /************************************checkpoint 回调选项******************************************
# 回调提供了几个选项，为 checkpoint 提供唯一名称并调整 checkpoint 频率。
#
# 训练一个新模型，每五个 epochs 保存一次唯一命名的 checkpoint ：

# # 在文件名中包含 epoch (使用 `str.format`)
# checkpoint_path = "./model/training_2/cp-{epoch:04d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# # 创建一个回调，每 5 个 epochs 保存模型的权重
# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path,
#     verbose=1,
#     save_weights_only=True,
#     period=5)

# model = create_model()
#
# # 使用 `checkpoint_path` 格式保存权重
# model.save_weights(checkpoint_path.format(epoch=0))


# # 使用新的回调*训练*模型
# model.fit(train_images,
#               train_labels,
#               epochs=50,
#               callbacks=[cp_callback],
#               validation_data=(test_images,test_labels),
#               verbose=0)
#查看模型保存的latest
# latest = tf.train.latest_checkpoint(checkpoint_dir)
# latest = 'model/training_2/cp-0050.ckpt'    #模型最后的路径


# #加载保存的参数权重 并重新进行衡量
# model = create_model()
# model.load_weights(latest)

#评估
# loss, acc= model.evaluate(test_images,test_labels,verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))  #Restored model, accuracy: 94.05%

# Checkpoints 包含：
#
# 一个或多个包含模型权重的分片。
# 索引文件，指示哪些权重存储在哪个分片中。


####################################手动保存权重################################
#手动保存权重
#
# history = model.fit(train_images,train_labels,batch_size=256,
#           epochs=10,
#           verbose=2,
#           validation_data=(test_images,test_labels))
#
# print(history.history.keys())
# # 保存权重

# # 创建模型实例
# model = create_model()
#
# # Restore the weights
# model.load_weights('./checkpoints/my_checkpoint')
#
# # Evaluate the model
# loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# ***************************将模型保存为HDF5文件**************************
# # 创建一个新的模型实例
# model = create_model()
#
# # 训练模型
# model.fit(train_images, train_labels, epochs=5)
#
# # 将整个模型保存为HDF5文件
# model.save('my_model.h5')

#Keras 可以使用 HDF5 标准提供基本保存格式。出于我们的目的，可以将保存的模型视为单个二进制blob


##************************** 重新创建完全相同的模型，包括其权重和优化程序**********************
new_model = keras.models.load_model('my_model.h5')

# 显示网络结构
new_model.summary()
#检查其准确率
loss, acc = new_model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# 这项技术可以保存一切:
# 1.权重
# 2.模型配置(结构)
# 3.优化器