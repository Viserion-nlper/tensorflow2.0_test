# 为了防止过度拟合，最好的解决方案是使用更完整的训练数据。数据集应涵盖模型应处理的所有输入范围。仅当涉及新的有趣案例时，其他数据才有用
# 经过更完整数据训练的模型自然会更好地推广。当这不再可能时.使用正则化会限制了模型可以存储的信息的数量和类型
# 则如果网络只能存储少量模式，那么优化过程将迫使它专注于最突出的模式，这些模式有更好的概括机会

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import regularizers

print(tf.__version__)


# import tensorflow_docs as tfdocs
# import tensorflow_docs.modeling
# import tensorflow_docs.plots
from  IPython import display
from matplotlib import pyplot as plt
# tf.compat.v1.enable_eager_execution()
import numpy as np
import pathlib
import shutil
import tempfile
logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)
gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')

FEATURES = 28
#tf.data.experimental.CsvDataset类可以被用来直接从没有中间减压工序的gzip文件读CSV记录
ds = tf.compat.v1.data.experimental.CsvDataset(gz,[float(),]*(FEATURES+1), compression_type="GZIP")

# 该csv阅读器类返回每个记录的标量列表。以下函数将标量列表重新打包为(feature_vector，label)对
def pack_row(*row):
  label = row[0]
  features = tf.stack(row[1:],1)
  return features, label

# 当处理大量数据时，TensorFlow效率最高。
#
# 因此，与其单独重新包装每一行，不如制作一个新的Dataset批次，该批次需要10000个示例，将pack_row函数应用于每个批次，然后将批次拆分回各个记录

packed_ds = ds.batch(10000).map(pack_row).unbatch()

# 查看一下值
for features,label in packed_ds.batch(1000).take(1):
  print(features[0])
  plt.hist(features.numpy().flatten(), bins = 101)
  plt.show()


N_VALIDATION = int(1e3)
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE

validate_ds = packed_ds.take(N_VALIDATION).cache()
train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()

validate_ds = validate_ds.batch(BATCH_SIZE)
train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=STEPS_PER_EPOCH*1000,
  decay_rate=1,
  staircase=False)

def get_optimizer():
  return tf.keras.optimizers.Adam(lr_schedule)

step = np.linspace(0,100000)
lr = lr_schedule(step)
plt.figure(figsize = (8,6))
plt.plot(step/STEPS_PER_EPOCH, lr)
plt.ylim([0,max(plt.ylim())])
plt.xlabel('Epoch')
_ = plt.ylabel('Learning Rate')

plt.show()



def get_callbacks(name):
  return [
    tfdocs.modeling.EpochDots(),
    tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
    tf.keras.callbacks.TensorBoard(logdir/name),
  ]

# L1正则化将权重推向正好为零，从而鼓励了稀疏模型。L2正则化将惩罚权重参数而不会使其稀疏，因为对于小权重，惩罚变为零。L2更常见的原因之一

# 在中tf.keras，通过将权重正则化器实例作为关键字参数传递给图层来添加权重正则化。现在添加L2权重正则化
l2_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001),
                 input_shape=(FEATURES,)),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1)
])

regularizer_histories['l2'] = compile_and_fit(l2_model, "regularizers/l2")

# l2(0.001)意味着该层权重矩阵中的每个系数都会增加网络0.001 * weight_coefficient_value**2的总损耗。
#
# 这就是为什么我们要binary_crossentropy直接监控。因为它没有混入此正则化组件。


# 添加dropout

# 最常用的神经网络正则化技术之一
# 直观解释是，由于网络中的各个节点不能依赖于其他节点的输出，因此每个节点必须输出自己有用的功能
# 在测试时，不会丢失任何单元，而是将图层的输出值按等于丢失率的比例缩小，以平衡活跃的单元比训练时更多的事实

dropout_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(1)
])

regularizer_histories['dropout'] = compile_and_fit(dropout_model, "regularizers/dropout")

#***************************************L2+dropOUT组合***********************************************
combined_model = tf.keras.Sequential([
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu', input_shape=(FEATURES,)),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(1)
])

regularizer_histories['combined'] = compile_and_fit(combined_model, "regularizers/combined")

# 带有"Combined"正则化的模型显然是迄今为止最好的模型。

#*******************************总结***************************************************

# 以下是防止神经网络过度拟合的最常用方法：
#
# 获取更多train数据。
# 减少网络容量。
# 添加weight regularization。权值正则化
# 添加dropout。

# 本文件未涵盖的两种重要方法是：
#
# 数据增加
# 批量标准化
# 请记住，每种方法都可以单独提供帮助，但通常将它们组合起来会更加有效。
# 如dropout + L2正则化

#****************************************************************************************