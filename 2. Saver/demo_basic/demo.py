# -*-coding:utf-8 -*-
import tensorflow as tf
import os

if __name__ == '__main__':
    train_dir = os.path.join('demo_model/', "demo")

    # with tf.device('/gpu:0'):
    a = tf.placeholder(dtype=tf.int32, shape=None, name='a')
    b = tf.placeholder(dtype=tf.int32, shape=None, name='b')
    y = tf.Variable(tf.ones(shape=[1], dtype=tf.int32), dtype=tf.int32, name='y')
    res = tf.add(tf.multiply(a, b), y, name='res')

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        feed_dict = {a:2, b:3}
        fetch_list = [res]
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        res = sess.run(feed_dict=feed_dict, fetches=fetch_list)
        saver.save(sess, train_dir)

        print("result: ", res[0])  # ('result: ', array([7], dtype=int32))
