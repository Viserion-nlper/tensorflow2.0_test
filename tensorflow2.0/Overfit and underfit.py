# 为了防止过度拟合，最好的解决方案是使用更完整的训练数据。数据集应涵盖模型应处理的所有输入范围。仅当涉及新的有趣案例时，其他数据才有用
# 经过更完整数据训练的模型自然会更好地推广。当这不再可能时.使用正则化会限制了模型可以存储的信息的数量和类型
# 则如果网络只能存储少量模式，那么优化过程将迫使它专注于最突出的模式，这些模式有更好的概括机会

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import regularizers

print(tf.__version__)



from  IPython import display
from matplotlib import pyplot as plt

import numpy as np

# import pathlib
# import shutil
# import tempfile
# logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
# shutil.rmtree(logdir, ignore_errors=True)
# gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')