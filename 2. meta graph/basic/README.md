




# 简介

把C++编译成可执行文件


本地mac上编译

```bash
g++ -std=c++11 demo.cc -g -o demo \
  -I /usr/local/include/eigen3 \
  -I /Library/Python/2.7/site-packages/tensorflow/include \
  -l tensorflow_cc \
  -l tensorflow_framework
```

linux服务器上编译

```bash

```

# freeze graph


```bash
TRAIN_DIR=model
freeze_graph \
    --input_meta_graph=${TRAIN_DIR}/demo.meta \
    --input_binary=true \
    --input_checkpoint=${TRAIN_DIR}/demo \
    --output_graph=${TRAIN_DIR}/frozen_graph.pb \
    --output_node_names=res
```

# 参考










# 参考 

https://www.jianshu.com/p/0c415b90404e