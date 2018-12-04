# -*- coding:utf-8 -*-
import tensorflow as tf

with tf.Session() as sess:
  # 1. restore graph
  saver = tf.train.import_meta_graph('model/my-model.meta.txt')
  # 或者 saver = tf.train.import_meta_graph('model/my-model.meta')

  # 2. restore checkpoint
  saver.restore(sess, 'model/my-model')
  # 或者 saver.restore(sess, tf.train.latest_checkpoint('demo_model/'))


  # 3. get tensor、variable、op
  logits = tf.get_default_graph().get_tensor_by_name('softmax_linear/logits:0')
  # 或者 logits = tf.get_collection("logits-from-collection")[0]
  print(sess.run(logits))


  # 4. Addes loss and train.
  labels = tf.constant(0, tf.int32, shape=[100], name="labels")
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                logits=logits)

  # Runs train_op.
  tf.summary.scalar('loss', loss)
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train_op = optimizer.minimize(loss)
  sess.run(train_op)