# -*- coding:utf-8 -*-
import tensorflow as tf


# restore
with tf.Session() as sess:
  saver = tf.train.import_meta_graph('model/my-model.meta')
  saver.restore(sess, 'model/my-model')

  graph = tf.get_default_graph()

  # get variable
  biases = graph.get_tensor_by_name('softmax_linear/biases:0')
  print(sess.run(biases))

  """ 失败的测试 get_variable
  通常都用 graph.get_tensor_by_name
  """
  # with tf.name_scope("hidden2"):
  #   weight = tf.get_variable('weights:0', shape=[128, 32])
  #   print(sess.run(weight))



  """ inference
  """
  # feed_dict = {a: 2, b: 3}
  # logits = tf.get_collection("logits")[0]
  logits = graph.get_tensor_by_name('softmax_linear/logits:0')
  print(sess.run(logits))


