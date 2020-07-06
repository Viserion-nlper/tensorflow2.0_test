import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


print(tf.__version__)

#加载clothesData

fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# #检查训练集中的第一张图像   colorbar颜色深浅条
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

#输入神经网络前要将其值控制在0-1范围内
train_images,test_images  = train_images/255.0, test_images/255.0


#显示训练集中的前25个图像，并在每个图像下方显示类名。
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.imshow(train_images[i])
#     plt.xlabel(class_names[train_labels[i]])
#     plt.xticks([])   #隐藏x y坐标的尺度标注
#     plt.yticks([])
# plt.show()


#神经网络的基本组成部分是层。图层从输入到其中的数据中提取表示。希望这些表示对于当前的问题有意义。
# 深度学习的大部分内容是将简单的层链接在一起。大多数层（例如tf.keras.layers.Dense）具有在训练过程中学习的参数

model  = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation="relu"),
    keras.layers.Dense(10,activation='softmax')
])
#第二层（也是最后一层）返回长度为10的logits数组。每个节点包含一个得分，该得分指示当前图像属于10类之一。



#损失函数 -衡量训练期间模型的准确性。您希望最小化此功能，以在正确的方向上“引导”模型。  计算差值损失
# 优化器 -这是基于模型看到的数据及其损失函数来更新模型的方式。  迭代
# 指标 -用于监视培训和测试步骤。以下示例使用precision，即正确分类的图像比例。
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


#然后开始feed Model
model.fit(train_images,train_labels,epochs=10)

#比较训练的模型在测试集上面表现 evaluate

#verbose：日志显示
# verbose = 0 为不在标准输出流输出日志信息
# verbose = 1 为输出进度条记录
# verbose = 2 为每个epoch输出一行记录
test_loss, test_acc = model.evaluate(test_images,test_labels,verbose=2)

print("test_loss",test_loss)
print("test_acc",test_acc)

#作出预测  predict
# 对数 (logits)
# 分类模型生成的原始（非标准化）预测向量，通常会传递给标准化函数。如果模型要解决多类别分类问题，则对数通常变成 softmax 函数的输入。
# 之后，softmax 函数会生成一个（标准化）概率向量，对应于每个可能的类别。


# 把model当作input添加到层中
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

# print(probability_model)
print("test_image_shape",test_images.shape)
predictions  = probability_model.predict(test_images)
#在这里，模型已经预测了测试集中每个图像的标签


#predictions.shape(10000,10)
# print(predictions[0])
# #输出[0.08536301 0.08536301 0.08536301 0.08536301 0.08536301 0.08554097 0.08536301 0.08536479 0.08536301 0.23155311]
# #接下来使用np.argmax进行筛选
# print(np.argmax(predictions[0]))
# print(test_labels[0])

#以图形方式查看完整的十个类预测
def plot_image(i,predictionArray,true_labels, img):
    predictionArray,true_label,img = predictionArray,true_labels[i],img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)

    prediction_label = np.argmax(predictionArray)
    print(predictionArray)
    if true_label == prediction_label:
        color = "blue"
    else:
        color = "red"
    plt.xlabel("{},{:2.0f}% ({})".format(class_names[prediction_label],100*np.max(predictionArray),class_names[true_label]), #这里定义np.max 而argmax是取索引
               color=color)
#画出概率条形图
def plot_value_array(i,predictionArray, true_labels):
    predictionArray,true_label = predictionArray,true_labels[i]
    plt.xticks(range(10))
    plt.yticks([])

    thisplot = plt.bar(range(10),predictionArray,color="#777777")
    plt.ylim([0,1])
    prediction_label_value = np.argmax(predictionArray)
    thisplot[prediction_label_value].set_color('red')  #设置预测标签为红色
    thisplot[true_label].set_color('blue')#设置真实标签为蓝色






# for i in range(20):
#     plt.figure(figsize=(6,3))
#     plt.subplot(1,2,1)
#     plot_image(i, predictions[i], test_labels, test_images)
#     plt.subplot(1,2,2)
#     plot_value_array(i,predictions[i],test_labels)
#     plt.show()

#设置5行6列的一个图表显示
num_rows = 5
num_cols = 3
num_images = num_cols * num_rows
plt.figure(figsize=(2*2*num_cols,2*num_rows))    #figsize的话 就设置其m行n列的两倍  设置figsize
for i in range(num_images):
    plt.subplot(num_rows,2*num_cols, i*2+1)  #subplot(m,n)  m行n列  第p个
    plot_image(i,predictions[i],test_labels,test_images)

    plt.subplot(num_rows,2*num_cols, i*2+2)
    plot_value_array(i,predictions[i],test_labels)
plt.tight_layout()
plt.show()

#对实际图片进行预测
img = test_images[1]
#需要对Img进行维度扩充  test_img.shape =（1000，28，28）
img = (np.expand_dims(img,0))

print(img.shape)  #(1, 28, 28)

pre_label_single = probability_model.predict(img)
print(pre_label_single)


# plot_value_array(1, pre_label_single[0], test_labels)
# _ = plt.xticks(range(10), class_names, rotation=45)