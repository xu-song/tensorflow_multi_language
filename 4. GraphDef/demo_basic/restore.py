# -*- coding:utf-8 -*-
import tensorflow as tf
import os
import numpy as np


train_dir = "model"
pb_file = os.path.join(train_dir, 'graph.pb')

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

if __name__ == '__main__':
    with tf.Session() as sess:
        graph = load_graph(pb_file)
        # saver.restore(sess, tf.train.latest_checkpoint('demo_model/'))
        # sess.run()
        a = graph.get_tensor_by_name("input/a:0")
        b = graph.get_tensor_by_name("input/b:0")
        feed_dict = {a: 2, b: np.array([[1,2],[3,4]])}

        op_to_restore = graph.get_tensor_by_name("res:0")
        print(sess.run(fetches=op_to_restore, feed_dict=feed_dict))


