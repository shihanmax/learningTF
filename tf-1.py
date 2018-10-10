"""
@ Author: mashihan
@ Date  : 2018/10/9 16:37
@ Use   : tf实现简单的字符识别
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession() # 该声明要放在tf.global_variables_initializer().run()之前

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

tf.global_variables_initializer().run()

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run(feed_dict={x:batch_xs, y_:batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels}))

"""
1. 定义算法公式（forward时的计算）
2. 定义loss，选定优化器，并指定优化器优化loss
3. 迭代地对数据进行训练
4. 在测试集或验证集上对准确率进行评测
"""