"""
@ Author: mashihan
@ Date  : 2018/10/11 18:00
@ Use   : tf实现多层卷积网络
"""
from models.tutorials.image.cifar10 import cifar10, cifar10_input
import tensorflow as tf
import numpy as np
import time

max_steps = 3000
batch_size = 128
data_dir = 'tmp/cifar10_data/cifar-10-batches-bin'


def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')  # wl来控制L2 loss的大小
        # 为weight增加l2正则化的loss，l1正则化会使权重变得稀疏（无用权重置0），l2正则化会防止特征权重过大。
        tf.add_to_collection("losses", weight_loss)  # 计算神经网络总体loss时用到

    return var


cifar10.maybe_download_and_extract()

# 生成训练数据（distorted_inputs用来执行数据增强）
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                            batch_size=batch_size)
# 生成测试数据，无须增强
images_test, labels_test = cifar10_input.inputs(eval_data=True,
                                                data_dir=data_dir,
                                                batch_size=batch_size)

image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])  # 这里的尺寸为什么不是[batch_size, 1000]

# 卷积层1
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2,
                                    wl=0.0)  # 5x5卷积核，64个，设置weights标准差0.05，第一个卷积层weight不正则，wl设置为0
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')  # 对image_holder卷积操作，步长均为1
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))  # 将bias初始化为0
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))  # 非线性层
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')  # 尺寸3x3.步长2x2最大池化，可以增加数据丰富性
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)  # 侧抑制机制，使响应值大的变得更大，来抑制其它神经元的影响，LRN适合ReLU这种没有上限的函数

# 卷积层2
weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')  # 这里调换了lrn和pooling层顺序

# fc层1
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

# fc层2
weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

# fc层3，不计算softmax
weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, wl=0.0)  # 注意此层的stddev设置
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5) + bias5)  # batch_size * 10


def loss(logits, labels):
    """
    计算total loss
    :param logits: 预测标签
    :param labels: 真实标签
    :return: total loss
    """
    labels = tf.cast(labels, tf.int64)
    # 以下计算softmax和cross entropy loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')

    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')  # 计算cross entropy的平均值
    tf.add_to_collection("losses", cross_entropy_mean)

    return tf.add_n(tf.get_collection("losses"), name='total_loss')  # 将所有loss求和


loss = loss(logits, label_holder)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)  # 计算结果中topk（这里为1）的预测结果

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()  # 数据增强的线程队列

for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])  # 获取一个batch的训练数据
    _, loss_value = sess.run([train_op, loss],
                             feed_dict={image_holder: image_batch, label_holder: label_batch})

    duraion = time.time() - start_time

    if step % 10 == 0:
        examples_per_sec = batch_size / duraion
        sec_per_batch = float(duraion)

        format_str = ('step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))

num_examples = 10000  # 测试集样本数量
import math
num_iter = int(math.ceil(num_examples / batch_size))  # 10000/batch_size 计算大概需要多少个iter将数据跑完
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch, label_holder: label_batch})

    true_count += np.sum(predictions)
    step += 1

precision = true_count / total_sample_count
print("precision @ 1 = %.3f" % precision)