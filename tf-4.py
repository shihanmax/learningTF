"""
@ Author: mashihan
@ Date  : 2018/10/10 14:07
@ Use   : tf实现单层卷积网络识别mnist手写数字
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)  # 由于使用ReLU，为避免产生死亡节点，为偏置加一个正值
    return tf.Variable(initial)


def conv2d(x, W):
    """
    2d卷积
    :param x: 输入
    :param W: 卷积参数，[size, size, channel, num of kernels]
    :return:卷积后的矩阵，与x同尺寸
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """
    最大池化
    :param x: 输入
    :return:池化后的矩阵
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 卷积层1，尺寸变化：28x28x1 -> 14x14x32
W_conv1 = weight_variable([5, 5, 1, 32])  # 卷积核尺寸5X5，1个颜色通道，32个卷积核
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 卷积层2，尺寸变化：14x14x32 -> 7x7x64
W_conv2 = weight_variable([5, 5, 32, 64])  # 这里的通道数是上一层的卷积核数量
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 全连接层1
W_fc1 = weight_variable([7 * 7 * 64, 1024])  # 全连接层节点数量1024
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
print(h_pool2_flat.shape)
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout层
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# softmax层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 定义损失
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(1e-4).minimize(cross_entropy)

# 定义正确率评估
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()
for i in range(20000):
    print('batch - {}'.format(i))
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_:batch[1], keep_prob: 1.0})
        print("step {}, training accuracy {}".format(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy {}".format(accuracy.eval(feed_dict={x: mnist.test.images,
                                                         y_: mnist.test.labels,
                                                         keep_prob: 1.0})))