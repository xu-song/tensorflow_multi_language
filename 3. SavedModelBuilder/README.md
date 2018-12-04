


# save in python



# 





# restore in python

# restore in java

java主要有两种方式，maven、JDK，以下采用JDK的方式介绍。

## 依赖

- libtensorflow.jar: 这只是个client，java API。底层实现仍然需要安装tensorflow
- tensorflow完整版，或者so。本例采用so


```bash
TF_VERSION="1.8.0"
TF_LIB=https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_VERSION}.jar
wget $TF_LIB

TF_TYPE="cpu" # Default processor is CPU. If you want GPU, set to "gpu"
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
mkdir -p ./jni
curl -L \
"https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-${TF_TYPE}-${OS}-x86_64-${TF_VERSION}.tar.gz" |
tar -xz -C ./jni
```


## Compiling

```bash
javac -cp libtensorflow-1.8.0.jar Restore.java
```

## Run

```bash
java -cp libtensorflow-1.8.0.jar:. -Djava.library.path=./jni Restore
```





# restore in C++

