from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf
import numpy as np
import pandas as pd
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

print(train.head())

train_y = train.pop('Species')
test_y = test.pop('Species')

# 标签列现已从数据中删除
print(train.head())

#******************************创建输入函数******************************************
def input_evaluation_set():
    features = {'SepalLength': np.array([6.4, 5.0]),
                'SepalWidth':  np.array([2.8, 2.3]),
                'PetalLength': np.array([5.6, 3.3]),
                'PetalWidth':  np.array([2.2, 1.0])}
    labels = np.array([2, 1])
    return features, labels


# 使用 Tensorflow 的 Dataset API，该 API 可以用来解析各种类型的数据。
#
# Dataset API 可以为您处理很多常见情况。例如，使用 Dataset API，您可以轻松地从大量文件中并行读取记录，并将它们合并为单个数据流。
#
# 为了简化此示例，我们将使用 pandas 加载数据，并利用此内存数据构建输入管道
def input_fn(features, labels, training=True, batch_size=256):
    """An input function for training or evaluating"""
    # 将输入转换为数据集。
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # 如果在训练模式下混淆并重复数据。
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)

#定义特征列（feature columns)
# 特征列（feature columns）是一个对象，用于描述模型应该如何使用特征字典中的原始输入数据。当您构建一个 Estimator 模型的时候，
# 会向其传递一个特征列的列表，其中包含您希望模型使用的每个特征。tf.feature_column 模块提供了许多为模型表示数据的选项。
#
# 对于鸢尾花问题，4 个原始特征是数值，因此我们将构建一个特征列的列表，以告知 Estimator 模型将 4 个特征都表示为 32 位浮点值

# 特征列描述了如何使用输入。
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

print(my_feature_columns)



#***************************实例化 Estimator****************************************
#
# tf.estimator.DNNClassifier 用于多类别分类的深度模型
# tf.estimator.DNNLinearCombinedClassifier 用于广度与深度模型
# tf.estimator.LinearClassifier 用于基于线性模型的分类器


# 构建一个拥有两个隐层，隐藏节点分别为 30 和 10 的深度神经网络。
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # 隐层所含结点数量分别为 30 和 10.
    hidden_units=[30, 10],
    # 模型必须从三个类别中做出选择。
    n_classes=3)


## 训练、评估和预测

# 已经有一个 Estimator 对象，现在可以调用方法来执行下列操作：
#
# 训练模型。
# 评估经过训练的模型。
# 使用经过训练的模型进行预测。

# ***********************************************训练模型 ******************************************。
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)

eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

#利用经过训练的模型进行预测（推理）

# 由模型生成预测
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}

def input_fn(features, batch_size=256):
    """An input function for prediction."""
    # 将输入转换为无标签数据集。
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

predictions = classifier.predict(
    input_fn=lambda: input_fn(predict_x))

for pred_dict, expec in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
        SPECIES[class_id], 100 * probability, expec))