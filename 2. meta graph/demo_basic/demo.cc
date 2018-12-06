//
// Created by MoMo on 17-8-10.
//
#include <iostream>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

using namespace std;
using namespace tensorflow;

int main()
{
    const string pathToGraph = "model/demo.meta";
    const string checkpointPath = "model/demo";

    SessionOptions options;
    options.config.set_allow_soft_placement(true);
    auto session = NewSession(options);

//    auto session = NewSession(SessionOptions());
    if (session == nullptr)
    {
        throw runtime_error("Could not create Tensorflow session.");
    }

    Status status;

// 读入我们预先定义好的模型的计算图的拓扑结构
    MetaGraphDef graph_def;
    status = ReadBinaryProto(Env::Default(), pathToGraph, &graph_def);
    if (!status.ok())
    {
        throw runtime_error("Error reading graph definition from " + pathToGraph + ": " + status.ToString());
    }

// 利用读入的模型的图的拓扑结构构建一个session
    status = session->Create(graph_def.graph_def());
    if (!status.ok())
    {
        throw runtime_error("Error creating graph: " + status.ToString());
    }

// 读入预先训练好的模型的权重
    Tensor checkpointPathTensor(DT_STRING, TensorShape());
    checkpointPathTensor.scalar<std::string>()() = checkpointPath;
    status = session->Run(
            {{ graph_def.saver_def().filename_tensor_name(), checkpointPathTensor },},
            {},
            {graph_def.saver_def().restore_op_name()},
            nullptr);
    if (!status.ok())
    {
        throw runtime_error("Error loading checkpoint from " + checkpointPath + ": " + status.ToString());
    }

    /*
    构造模型的输入，相当与python版本中的feed
        a = graph.get_tensor_by_name("input/a:0")
        b = graph.get_tensor_by_name("input/b:0")
        feed_dict = {a: 2, b: 3}
     */
    std::vector<std::pair<string, Tensor>> input;
    Tensor a(tensorflow::DT_INT32, tensorflow::TensorShape({1}));
    Tensor b(tensorflow::DT_INT32, tensorflow::TensorShape({1}));

    auto a_map = a.tensor<int,1>();
    a_map(0) = 2;
    auto b_map = b.tensor<int,1>();
    b_map(0) = 3;
    // b采用矩阵，怎样用？ b_map=b.tensor<int,2>(); b_map(0)={{3, 2}, {-1, 0}};
    input.emplace_back(std::string("a"), a);
    input.emplace_back(std::string("b"), b);

//   运行模型，并获取输出
    std::vector<tensorflow::Tensor> answer;
    status = session->Run(input, {"res"}, {}, &answer);

    Tensor result = answer[0];  // debug的时候，p answer[0] 为什么显示 error: Couldn't lookup symbols:
    auto result_map = result.tensor<int,1>();  // 1*n 的向量，也包括1*1
    cout<<"result: "<<result_map(0)<<endl;

    return 0;
}

