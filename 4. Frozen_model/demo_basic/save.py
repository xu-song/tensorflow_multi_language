# -*-coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import sys, os
import numpy as np


train_dir = "model"

if __name__ == '__main__':
    with tf.variable_scope('input') as scope:
        a = tf.placeholder(dtype=tf.int32, shape=None, name='a')
        b = tf.placeholder(dtype=tf.int32, shape=[2, 2], name='b')

    y = tf.Variable(tf.ones(shape=[1], dtype=tf.int32), dtype=tf.int32, name='y')
    res = tf.add(tf.multiply(a, b), y, name='res')  # res = a*b + 1
    with tf.Session() as sess:
        feed_dict = dict()
        feed_dict[a] = 2
        feed_dict[b] = np.array([[1, 2], [3, 4]])

        fetch_list = [res]
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        # 训练和保存模型
        res = sess.run(feed_dict=feed_dict, fetches=fetch_list)
        saver.save(sess, os.path.join(train_dir, "ckpt"))  # 默认参数: write_meta_graph=True. 调用export_meta_graph
        saver.export_meta_graph(os.path.join(train_dir,'ckpt.meta.txt'), as_text=True)  # 不能用于freeze graph
        tf.train.write_graph(sess.graph_def, train_dir, "graph.pbtxt", as_text=True)    # 可以freeze graph
        tf.train.write_graph(sess.graph_def, train_dir, "graph.pb", as_text=False)

        print("result: ", res[0])
