from argparse import ArgumentParser

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.tools import optimize_for_inference_lib

import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# First, we replace Mean and Squeeze nodes in the ResNet50 graph by
# AvgPool and Reshape, respectively. We store the resulting GraphDef
# object in `combined_gdef` variable with float32 precision.
gdef_file = "resnet50_v1.pb" 
input_gdef = graph_pb2.GraphDef()
with open(gdef_file, "rb") as f:
  data = f.read()
  input_gdef.ParseFromString(data)

input_names_first_subgraph = ["input_tensor"]
output_names_first_subgraph = ["resnet_model/Relu_48"]
input_names_second_subgraph = ["resnet_model/Squeeze"]
output_names_second_subgraph = ["ArgMax", "softmax_tensor"]

first_gdef = optimize_for_inference_lib.optimize_for_inference(
    input_gdef, input_names_first_subgraph, output_names_first_subgraph, tf.float32.as_datatype_enum)
second_gdef = optimize_for_inference_lib.optimize_for_inference(
    input_gdef, input_names_second_subgraph, output_names_second_subgraph, tf.float32.as_datatype_enum)

g = tf.Graph()
with g.as_default():
  tf.compat.v1.import_graph_def(first_gdef, name='')
  last_relu = g.get_tensor_by_name('resnet_model/Relu_48:0')
  mean = tf.nn.avg_pool(last_relu, [1, 7, 7, 1], [1, 1, 1, 1], 'VALID', data_format='NHWC', name='resnet_model/Mean')
  squeeze = tf.reshape(mean, [-1, 2048], name='resnet_model/Squeeze')

combined_gdef = g.as_graph_def()
for node in second_gdef.node:
  if node.name == 'resnet_model/Squeeze':
    continue
  else:
    combined_gdef.node.append(node)

# Second, we convert float32 precision to bfloat16 precision using
# auto-mixed-precision graph optimizer from TensorFlow. The resulting GraphDef
# is stored in `optimized_gdef` variable.
g = tf.Graph()
with g.as_default():
  importer.import_graph_def(combined_gdef, input_map={}, name="")
  meta_graph = saver_lib.export_meta_graph(graph_def=combined_gdef, graph=g)

  fetch_collection = meta_graph_pb2.CollectionDef()
  for fetch in output_names_second_subgraph:
    fetch_collection.node_list.value.append(fetch)
  meta_graph.collection_def["train_op"].CopyFrom(fetch_collection)

config = config_pb2.ConfigProto()
config.graph_options.rewrite_options.CopyFrom(
    rewriter_config_pb2.RewriterConfig(
    auto_mixed_precision_mkl=rewriter_config_pb2.RewriterConfig.ON,
    remapping=rewriter_config_pb2.RewriterConfig.OFF))
optimized_gdef = tf_optimizer.OptimizeGraph(
    config, meta_graph)

with open("bf16_resnet50_v1.pb", "wb") as f:
  f.write(optimized_gdef.SerializeToString())
