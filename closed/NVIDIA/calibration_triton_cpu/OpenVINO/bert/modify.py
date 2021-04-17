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

model = onnx.load("bert_large_v1_1_fake_quant.onnx")
graph = model.graph
input0 =  graph.input[0]
input1 =  graph.input[1]
input2 =  graph.input[2]
input0.type.tensor_type.elem_type = 6 #int32
input1.type.tensor_type.elem_type = 6
input2.type.tensor_type.elem_type = 6
graph.input.remove(graph.input[2])
graph.input.remove(graph.input[1])
graph.input.remove(graph.input[0])
graph.input.extend([input0])
graph.input.extend([input1])
graph.input.extend([input2])
for n in graph.node:
    if "Gather" in n.name:
        print(n)
onnx.save(model, "bert_large_v1_1_fake_quant_int32.onnx")
