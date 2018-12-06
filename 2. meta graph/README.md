



# 教程1

- [Tensorflow C++ API调用Python预训练模型 | 简书](1. Saver/demo)



# 教程2




# 教程3


SavedModelBuilder = export_meta_graph + save_variable
saver.save = export_meta_graph + save_ch



```yml
tf.get_default_graph().get_tensor_by_name( : 是在graph上获取
tf.get_collection: 与graph无关
tf.get_variable  跟get_tensor_by_name什么区别？
```

## 教程 xusong

搭配用法

tf.train.import_meta_graph 只能与 saver.export_meta_graph 搭配使用。
不能和 tf.train.write_graph 使用。


# restore from C++


## 编译成可执行文件

参考 https://www.jianshu.com/p/0c415b90404e
