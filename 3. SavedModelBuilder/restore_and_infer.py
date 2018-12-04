# -*- coding:utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os

mnist = input_data.read_data_sets('/tmp/data/MNIST_data', one_hot=True)

# pb模型的恢复
export_dir = os.path.dirname(os.path.abspath(__file__)) + '/model'
def restore_model_pb():
    sess = tf.Session()
    tf.saved_model.loader.load(sess, ['mytag'], export_dir)


    # get weight
    w = sess.graph.get_tensor_by_name('w:0')
    print('params: w = ', sess.run(w))


    input_x = sess.graph.get_tensor_by_name('input_x:0')  # place_holder
    op = sess.graph.get_tensor_by_name('predict:0')
    print(sess.run(op, feed_dict={input_x: np.expand_dims(mnist.test.images[15], axis=0)}))
    sess.close()


restore_model_pb()