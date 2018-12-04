# coding: utf-8
""" A simple script for inspect checkpoint files."""

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


f_path = '/Users/bitmain/tmp/projector/oss_data/word2vec_10000_200d_tensors.bytes'  # 这到底是什么格式？不符合ckp的格式
'''
报错: Data loss: not an sstable (bad magic number): perhaps your file is in a different file format 
ckpt的magic number是什么？
'''


# f_path = 'model/ckpt'

print_tensors_in_checkpoint_file(
    f_path,
    True,
    True)

'''
该代码用于读取 V2 format的checkpoint，f_path不要加尾部扩展.

如何读取V1的checkpoint

'''