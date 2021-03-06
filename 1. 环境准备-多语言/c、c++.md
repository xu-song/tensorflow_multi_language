


# 必读

tf各个版本的环境兼容: https://www.tensorflow.org/install/source#common_installation_problems

## 下载tensorflow源码

```
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
git checkout r1.8
```

## 编译tensorflow源码

### CPU-only

```sh
./configure
# 把tensorflow源码编译成whl文件，编译后保存在 bazel-bin/tensorflow/libtensorflow.so
bazel build --config=opt //tensorflow:libtensorflow_cc.so
```

### GPU-support

```sh
./configure
# 编译C API的库
bazel build --config=opt --config=cuda //tensorflow:libtensorflow.so

# 或者 编译C++ API的库
bazel build --config=opt --config=cuda //tensorflow:libtensorflow_cc.so
```

## 配置环境

拷贝到全局环境中，或者设定环境变量。

```sh
# 拷贝到全局环境中
TF_PATH=/usr/local/include/  # 也可以指定其他路径
sudo cp -r bazel-genfiles/* ${TF_PATH}
sudo cp -r tensorflow/c ${TF_PATH}tensorflow/
sudo cp -r tensorflow/cc ${TF_PATH}tensorflow/
sudo cp -r tensorflow/core ${TF_PATH}tensorflow/
sudo cp -r third_party ${TF_PATH}
sudo cp -r bazel-bin/tensorflow/libtensorflow_cc.so $LIB_PATH
sudo cp -r bazel-bin/tensorflow/libtensorflow_framework.so $TF_PATH
```

# 编译

**本地mac**

```sh
g++ -std=c++11 -I /usr/local/include/eigen3 \
  -I /Library/Python/2.7/site-packages/tensorflow/include \
  -l tensorflow_cc \
  -l tensorflow_framework \
  example.cc -o example
```



**linux服务器**


```bash

export LD_LIBRARY_PATH=/root/tensorflow/bazel-bin/tensorflow:$LD_LIBRARY_PATH


g++ -std=c++11 hello_tf.cc -o example  \
  -I /root/tensorflow/bazel-genfiles \
  -I /root/tensorflow  \
  -I /usr/local/lib/python2.7/dist-packages/tensorflow/include/external/eigen_archive \
  -I /usr/local/lib/python2.7/dist-packages/tensorflow/include \
  -L /root/tensorflow/bazel-bin/tensorflow  \
  -l tensorflow_cc \
  -l tensorflow_framework


./hello_tf

```


### troble shooting


**Eigen/CXX11/Tensor: No such file or directory**

https://github.com/tensorflow/tensorflow/issues/13705

**This file was generated by an older version of protoc**

一般是因为，安装了多个protobuf版本，编译的时候采用了错误的版本。

有多个版本
- pip 安装的
    - pip show protobuf
- apt-get install 安装的
    - protoc --version
- tensorflow自带的

版本号
3004000 代表3.4.0
3004001 代表3.4.1  
3005000
