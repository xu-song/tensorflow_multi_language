# -*- coding:utf-8 -*-
'''
by chen xianling
'''
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os


mnist = input_data.read_data_sets('/tmp/data/MNIST_data', one_hot=True)


x = tf.placeholder("float", shape=[None, 784], name='input_x')  # 输入图像占位符
y_ = tf.placeholder("float", shape=[None, 10])  # 标签类别占位符

W = tf.Variable(tf.zeros([784, 10]), name='w')  # 权重W是一个784x10的矩阵（因为我们有784个特征和10个输出值）
b = tf.Variable(tf.zeros([10]), name='b')
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

y = tf.nn.softmax(tf.matmul(x, W) + b, name='predict')

cross_entropy = - tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


for i in range(1000):
    batch = mnist.train.next_batch(50)  # 每一步迭代加载50个训练样本，然后执行一次train_step
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
    if i % 100 == 0:
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))  # 模型在测试数据集上面的正确率

# print(mnist.test.images[15])
print(sess.run(y, feed_dict={x: np.expand_dims(mnist.test.images[15], axis=0)}))
print(sess.run(tf.argmax(sess.run(y, feed_dict={x: np.expand_dims(mnist.test.images[15], axis=0)}), axis=1)))
print(mnist.test.labels[15])



# 保存 SavedModel
# 参考 https://www.tensorflow.org/guide/saved_model#manually_build_a_savedmodel
export_dir = os.path.dirname(os.path.abspath(__file__)) + '/model'
builder = tf.saved_model.builder.SavedModelBuilder(export_dir)  # 声明一个空的builder
builder.add_meta_graph_and_variables(sess, ["mytag"])   # 保存graph到.pb，保存变量到
# builder.save()       # graph存为pb格式，
builder.save(as_text=True)  # 只会把graph存储为pbtxt格式，variable格式不变
sess.close()

