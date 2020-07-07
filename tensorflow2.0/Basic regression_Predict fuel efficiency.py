from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print(dataset_path)

# 我们在使用pandas读取文件时，常会遇到某个字段为NaN。
#
# 一般情况下，这时因为文件中包含空值导致的，因为pandas默认会将
#
# '-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A', '#N/A', 'N/A', 'NA', '#NA', 'NULL', 'NaN', '-NaN', 'nan', '-nan', ''
# 判定为缺失值，从而转换为NaN。
# 那么如何避免DATa
# Frame中出现NaN呢，使用keep_default_na参数可以解决。
# keep_default_na参数用来控制是否要将被判定的缺失值转换为NaN这一过程，默认为True。，当keep_default_na = False时，源文件中出现的什么值，DataFrame中就是什么值。
#
#
# 下来再说na_values参数， 这个参数用来控制那些值会被判定为缺失值，它接收一个列表或者集合，当列表或者几个中出现的字符串在文件中出现时，它也会被判定为缺失值.
# 但是，无论此时keep_default_na = True还是False，他都将被改写。


column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)  #参数搞一下

dataset = raw_dataset.copy()
dataset.tail()
# print(dataset.tail())

#数据清洗
dataset.isna().sum()

# 为了保证这个初始示例的简单性，删除这些行。
dataset = dataset.dropna()

# print(dataset)
origin = dataset.pop('Origin')
# print(origin)

dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
# print(dataset.tail())  #将origin变成了one-hot编码
#       MPG  Cylinders  Displacement  Horsepower  ...  Model Year  USA  Europe  Japan
# 393  27.0          4         140.0        86.0  ...          82  1.0     0.0    0.0
# 394  44.0          4          97.0        52.0  ...          82  0.0     1.0    0.0
# 395  32.0          4         135.0        84.0  ...          82  1.0     0.0    0.0
# 396  28.0          4         120.0        79.0  ...          82  1.0     0.0    0.0
# 397  31.0          4         119.0        82.0  ...          82  1.0     0.0    0.0


#拆分训练数据集和测试数据集
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset  = dataset.drop(train_dataset.index)  #dropXXXindex
print("test_dataset",test_dataset.shape)

#查看sns关系图
# sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
# plt.show()


#也可以查看总体的数据统计:
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
# print(train_stats)

#将特征值从目标值或者"标签"中分离。 这个标签是使用训练模型进行预测的值。
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')


#做归一化处理  用于归一化输入的数据统计（均值和标准差）
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

#      Cylinders  Displacement  Horsepower  ...       USA    Europe     Japan
# 146  -0.869348     -1.009459   -0.784052  ...  0.774676 -0.465148 -0.495225
# 282  -0.869348     -0.530218   -0.442811  ...  0.774676 -0.465148 -0.495225

# print(normed_train_data)
print(len(train_dataset.keys()))  #9 列数据

def build_model():
    model = keras.Sequential([
    layers.Dense(64,activation="relu",input_shape=[len(train_dataset.keys())]),
    layers.Dense(64,activation="relu"),
    layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


model = build_model()
model.summary()

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
# print(example_result)


#对模型进行1000个周期的训练，并在 history 对象中记录训练和验证的准确性。
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('*')
    print('.', end='')
EPOCHS = 1000

# history = model.fit(normed_train_data,train_labels,
#           epochs=EPOCHS,
#           verbose=0,
#           validation_split=0.2,
#           callbacks=[PrintDot()])

#********************************添加了early_stop的model.fit***********************************
model = build_model()

# 图表显示在约100个 epochs 之后误差非但没有改进，反而出现恶化。 让我们更新 model.fit 调用，当验证值没有提高上是自动停止训练
# 使用一个 EarlyStopping callback 来测试每个 epoch 的训练条件
# 若经过一定数量的 epochs 后没有改进，则自动停止训练。
# 图表显示在约100个 epochs 之后误差非但没有改进，反而出现恶化。 让我们更新 model.fit 调用，当验证值没有提高上是自动停止训练
# 使用一个 EarlyStopping callback 来测试每个 epoch 的训练条件
# 若经过一定数量的 epochs 后没有改进，则自动停止训练。
# ***************************************参数*************************************************
# monitor: 监控的数据接口，有’acc’,’val_acc’,’loss’,’val_loss’等等。正常情况下如果有验证集，就用’val_acc’或者’val_loss’
# min_delta：增大或减小的阈值，只有大于这个部分才算作improvement。这个值的大小取决于monitor，也反映了你的容忍程度。例如笔者的monitor是’acc’，同时其变化范围在70%-90%之间，所以对于小于0.01%的变化不关心。加上观察到训练过程中存在抖动的情况（即先下降后上升），所以适当增大容忍程度，最终设为0.003%
# patience：能够容忍多少个epoch内都没有improvement。
# mode: 就’auto’, ‘min’, ‘,max’三个可能。如果知道是要上升还是下降，建议设置一下。
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',patience=10)
history  = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop,PrintDot()])

# 使用 history 对象中存储的统计信息可视化模型的训练进度。
# print("  ")
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.keys())  #Index(['loss', 'mean_absolute_error', 'mean_squared_error', 'val_loss','val_mean_absolute_error', 'val_mean_squared_error', 'epoch'],

#            loss  mean_absolute_error  ...  val_mean_squared_error  epoch
# 0    556.569112            22.287327  ...              543.604065      0
# 1    499.769846            20.969002  ...              488.904327      1
# 2    447.287187            19.701649  ...              432.344238      2
# 3    391.477367            18.282650  ...              370.361237      3
# 4    332.447819            16.658438  ...              307.165314      4

# def plot_history(history):
#   hist = pd.DataFrame(history.history)
#   hist['epoch'] = history.epoch
#
#   plt.figure()
#   plt.xlabel('Epoch')
#   plt.ylabel('Mean Abs Error [MPG]')
#   plt.plot(hist['epoch'], hist['mae'],
#            label='Train Error')
#   plt.plot(hist['epoch'], hist['val_mae'],
#            label = 'Val Error')
#   plt.ylim([0,5])
#   plt.legend()


#绘制平均绝对误差图表MAE
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("mean abs Error")
    plt.plot(hist['epoch'],hist['mean_absolute_error'],
             label="Train Error")

    plt.plot(hist['epoch'],hist['val_mean_absolute_error'],
             label="Val Error")

    plt.ylim([0,5])
    plt.legend()
    plt.show()


#绘制均方误差
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()
    plt.show()

#
# plot_history(history)




plot_history(history)

#看看通过使用 测试集 来泛化模型的效果如何
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

# print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

#最后，使用测试集中的数据预测 MPG 值:

test_predictions = model.predict(normed_test_data)
# print(test_predictions.shape)   #(78, 1)

test_predictions = test_predictions.flatten()
print(test_predictions)

# 测试集中的数据预测 MPG 值:
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
print("aaa",plt.xlim()[1])  #aaa 46.065  就是x轴的最大值
plt.xlim([0,plt.xlim()[1]])  #设定范围
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([0, 100], [0, 100])
plt.show()

#看下误差分布
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()


#总结回归分析中的知识点：
# 均方误差（MSE）是用于回归问题的常见损失函数（分类问题中使用不同的损失函数
# 常见的回归指标是平均绝对误差（MAE）
# 当数字输入数据特征的值存在不同范围时，每个特征应独立缩放到相同范围
# 如果训练数据不多，一种方法是选择隐藏层较少的小网络，以避免过度拟合
# 早期停止是一种防止过度拟合的有效技术