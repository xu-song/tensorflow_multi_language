
/*
参考
1. https://www.tensorflow.org/guide/saved_model#load_a_savedmodel_in_c
2. https://www.jianshu.com/p/0c415b90404e

*/


#include <fstream>
#include <utility>
#include <vector>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;


int main(int argc, char* argv[]) {
    const string export_dir = "model/";

    // load a SavedModel
    SavedModelBundle bundle;
    LoadSavedModel(session_options, run_options, "./model", {kSavedModelTagTrain},
                   &bundle);

    SessionOptions options;
    std::unique_ptr<Session> session(NewSession(options));

    return 1;

}

