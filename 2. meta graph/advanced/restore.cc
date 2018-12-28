

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
    const string pathToGraph = "model/my-model.meta";
    const string checkpointPath = "model/my-model";
    auto session = NewSession(SessionOptions());
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


//   运行模型，并获取输出
    std::vector<tensorflow::Tensor> answer;
    status = session->Run({}, {"softmax_linear/logits:0"}, {}, &answer);

    Tensor result = answer[0];
    auto result_map = result.tensor<int,3>();
    cout<<"result: "<<result_map(0)<<endl;

    return 0;
}
