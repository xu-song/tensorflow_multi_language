# -*- coding:utf-8 -*-
import tensorflow as tf
import math

# Creates an inference graph.
# Hidden 1
images = tf.constant(1.2, tf.float32, shape=[100, 28])
with tf.name_scope("hidden1"):
    weights = tf.Variable(tf.truncated_normal([28, 128], stddev=1.0 / math.sqrt(float(28))), name="weights")
    biases = tf.Variable(tf.zeros([128]), name="biases")
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
# Hidden 2
with tf.name_scope("hidden2"):
    weights = tf.Variable(tf.truncated_normal([128, 32], stddev=1.0 / math.sqrt(float(128))), name="weights")
    biases = tf.Variable(tf.zeros([32]), name="biases")
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

# Linear
with tf.name_scope("softmax_linear"):
    weights = tf.Variable(tf.truncated_normal([32, 10], stddev=1.0 / math.sqrt(float(32))), name="weights")
    biases = tf.Variable(tf.zeros([10]), name="biases")
    tf.summary.scalar('test_summary', tf.reduce_sum(biases))
    # logits = tf.matmul(hidden2, weights) + biases  # 这里可以加个name
    logits = tf.add(tf.matmul(hidden2, weights), biases, name="logits")
    print('name: %s' % str(logits.name))
    print('shape: %s' % str(logits.shape))
    tf.add_to_collection("logits-from-collection", logits)   # 把变量放入一个集合，后面可以从集合中取出

init_all_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # Initializes all the variables.
    sess.run(init_all_op)
    # Runs to logit.
    sess.run(logits)
    # Creates a saver.
    saver0 = tf.train.Saver()

    """ 1. 保存graph和checkpoint (建议)
      这里会保存graph(.meta)和checkpoint
      看各种教程不如看源码：
      write_meta_graph=True: 调用export_meta_graph
      write_state=True,
    """
    saver0.save(sess, 'model/my-model')

    """ 2.1 只保存graph
      这种方式进行freeze graph，需要加入参数input_meta_graph
    """
    # Generates MetaGraphDef.
    # 方式一：
    saver0.export_meta_graph('model/my-model.meta')  # 只保存一个graph文件、
    saver0.export_meta_graph('model/my-model.meta.txt', as_text=True)

    """ 2.2 只保存graph
       对应 tf.train.import_meta_graph
    """
    tf.train.export_meta_graph("model/fdsa.txt")

    """ 2.3：只保存graph
      这种graph采用 tf.train.import_meta_graph会报错
    """
    tf.train.write_graph(sess.graph_def, 'model/', "my-model.pbtxt", as_text=True)
    tf.train.write_graph(sess.graph_def, 'model/', "my-model.pb", as_text=False)

    # summary for tensorboard

    for i in range(10):
        tf.summary.scalar('test_summary', i)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('model/', sess.graph)
    test_writer = tf.summary.FileWriter('model/')
