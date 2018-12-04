

# Tensorflow C++ 编译和调用图模型

博客: https://blog.csdn.net/rockingdingo/article/details/75452711
github: https://github.com/rockingdingo/tensorflow-tutorial



# 遇到的问题

## freeze graph的过程

freeze_graph之后的文件很小，没有上面说的那么大。



## 编译C++ 的过程

```bash
g++ -std=c++11 -I /usr/local/include/eigen3 \
  -I /Library/Python/2.7/site-packages/tensorflow/include \
  -l tensorflow_cc \
  -l tensorflow_framework \
  ./src/ann_model_loader.cpp \
  -o ./bin/ann_model_loader.o
  
  
/usr/bin/g++-5 -c  -std=c++11   

/usr/bin/g++-5 -c -std=c++11 -o ./bin/main.o -I/home/songbai.pu/software/protobuf-3.6.0/src -I/usr/local/tf/include -I/usr/local/tf-1.2/include/ori -I/usr/local/include/eigen3 -g  ./src/main.cpp 

/usr/bin/g++-5 -o bin/test bin/main.o  bin/ann_model_loader.o /usr/local/tf/lib/libtensorflow_cc.so /usr/local/tf/lib/libtensorflow_framework.so



folder_dir=`pwd`
model_path=${folder_dir}/model/nn_model_frozen.pb

#cp binary to root folder
cp ./bin/test ./tfcpp_demo

./tfcpp_demo ${model_path}
```