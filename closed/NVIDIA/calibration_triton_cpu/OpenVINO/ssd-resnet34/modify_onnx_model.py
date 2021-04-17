# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import onnx
import numpy as np
from onnx import TensorProto
model = onnx.load("resnet34-ssd1200.onnx")
graph = model.graph
shape = np.array([1, 200, 1], dtype=np.int64)
for i in range(len(graph.node)):
    if "scores" in graph.node[i].output:
        print(graph.node[i])
        node = onnx.helper.make_node('Unsqueeze',inputs=['scores'],outputs=['scores_expanded'],axes=[2])
        graph.node.insert(i+1,node)
for i in range(len(graph.node)):
    if "labels" in graph.node[i].output:
        print(graph.node[i])
        node = onnx.helper.make_node('Cast',inputs=['labels'],outputs=['labels_fp32'],to=getattr(TensorProto, 'FLOAT'))
        graph.node.insert(i+1,node)
for i in range(len(graph.node)):
    if "labels_fp32" in graph.node[i].output:
        print(graph.node[i])
        node = onnx.helper.make_node('Unsqueeze',inputs=['labels_fp32'],outputs=['labels_expanded'],axes=[2])
        graph.node.insert(i+1,node)

node = onnx.helper.make_node('Concat',name="Concat_999",inputs=['bboxes','labels_expanded','scores_expanded','scores_expanded'],outputs=['output'],axis=2)
graph.node.append(node)
ovi = onnx.helper.make_tensor_value_info("output",elem_type=1,shape=[1,"nbox",7])
graph.output.append(ovi)
onnx.checker.check_model(model)
onnx.save(model,"resnet34-ssd1200_modified.onnx")
