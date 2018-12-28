# coding: utf-8
"""
restore by TensorRT
"""

# Import TensorRT Modules
import tensorrt as trt
import uff
from tensorrt.parsers import uffparser
G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)

# Load your newly created Tensorflow frozen model and convert it to UFF
uff_model = uff.from_tensorflow_frozen_model("model/frozen_graph.pb", ["res"])

# Create a UFF parser to parse the UFF file created from your TF Frozen model
parser = uffparser.create_uff_parser()
parser.register_input("input_1", (3,224,224),0)
parser.register_output("res")


# Build your TensorRT inference engine# This step performs (1) Tensor fusion (2) Reduced precision# (3) Target autotuning (4) Tensor memory management
engine = trt.utils.uff_to_trt_engine(G_LOGGER,
                                     uff_model,
                                     parser,
                                     1,
                                     1<<20,
                                     trt.infer.DataType.FLOAT)

# Serialize TensorRT engine to a file for when you are ready to deploy your model.
trt.utils.write_engine_to_file("model/demo.engine", engine.serialize())