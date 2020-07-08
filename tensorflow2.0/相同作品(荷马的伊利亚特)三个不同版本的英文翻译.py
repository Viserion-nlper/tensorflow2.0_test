from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow_datasets as tfds  #需要2.0版本

import tensorflow as tf

import os

#使用的文本文件已经进行过一些典型的预处理，主要包括删除了文档页眉和页脚，行号，章节标题。


DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']

# for name in FILE_NAMES:
#     text_dir = tf.keras.utils.get_file(name, origin=DIRECTORY_URL + name)
#
# parent_dir = os.path.dirname(text_dir)
#
# print(parent_dir)
parent_dir ="./datasets"
def labeler(example, index):
  return example, tf.cast(index, tf.int64)

labeled_data_sets = []

for i, file_name in enumerate(FILE_NAMES):
  print(os.path.join(parent_dir, file_name))
  lines_dataset = tf.data.TextLineDataset(os.path.join(parent_dir, file_name))
  print(lines_dataset)
  labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
  labeled_data_sets.append(labeled_dataset)

print(labeled_data_sets)


BUFFER_SIZE = 50000
BATCH_SIZE = 64
TAKE_SIZE = 5000

all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
    all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

all_labeled_data = all_labeled_data.shuffle(
    BUFFER_SIZE, reshuffle_each_iteration=False)

for ex in all_labeled_data.take(5):
  print(ex)


#将文本编码成数字

#建立词汇表  1.通过将文本标记为单独的单词集合来构建词汇表
# 迭代每个样本的 numpy 值。
# 使用 tfds.features.text.Tokenizer 来将其分割成 token。
# 将这些 token 放入一个 Python 集合中，借此来清除重复项。
# 获取该词汇表的大小以便于以后使用

tokenizer = tfds.features.text.Tokenizer()
vocabulary_set = set()
for text_tensor, _ in all_labeled_data:
    # print(text_tensor)
    # print(text_tensor.numpy())
    tokenizer.tokenize(text_tensor.numpy())
    some_tokens = tokenizer.tokenize(text_tensor.numpy())
    # print(some_tokens)
    vocabulary_set.update(some_tokens)

vocab_size = len(vocabulary_set)

# print(vocab_size)
#样本编码
# 通过传递 vocabulary_set 到 tfds.features.text.TokenTextEncoder 来构建一个编码器。
# 编码器的 encode 方法传入一行文本，返回一个整数列表。

encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)


#查看输出原始样式  b'And his own comrades from the circus forth'
example_text = next(iter(all_labeled_data))[0].numpy()
print(example_text)


# 进行encoder转换。  #利用了词袋模型 讲文本对序列一一对应
encoded_example = encoder.encode(example_text)
print(encoded_example)




