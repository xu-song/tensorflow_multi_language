
# -*- coding:utf-8 -*-
import tensorflow as tf
import os
import numpy as np

if __name__ == '__main__':
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph('demo_model/demo.meta')
        saver.restore(sess, tf.train.latest_checkpoint('demo_model/'))
        graph = tf.get_default_graph()
        a = graph.get_tensor_by_name("a:0")
        b = graph.get_tensor_by_name("b:0")
        feed_dict = {a: 2, b: 3}

        op_to_restore = graph.get_tensor_by_name("res:0")
        print(sess.run(fetches=op_to_restore, feed_dict=feed_dict))


