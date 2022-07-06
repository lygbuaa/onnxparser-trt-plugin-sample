#!/usr/bin/env python3
#
# Copyright 1993-2020 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.

import onnxruntime
import numpy as np
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import common
import torch
import torch.nn.functional as F
import torch.onnx.symbolic_helper as sym_help
import onnx_graphsurgeon as gs
import onnx
from custom_grid_sample import custom_grid_sample


'''
this samples demonstrates:
1. exporting custom op in torch to onnx (with dynamic shape support, specificially, explicit batch)
2. parsing onnx model to tensorrt using python api with plugin support
3. dynamic shape python API test. -1 for network input batch and 2 for runtime context batch.

source files:
test_plugn_result.py   # main
symbolic_opset10.py    # onnx custom op patch
common.py              # modified allocate_buffers() function to support explicit batch.
'''

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

trt.init_libnvinfer_plugins(TRT_LOGGER, '')
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list
for plugin_creator in PLUGIN_CREATORS:
    print("find plugin_creator: {}".format(plugin_creator.name))
# print("PLUGIN_CREATORS: {}".format(PLUGIN_CREATORS))

input_rand = np.random.rand(4, 1, 4, 4).astype('float32')
grid_rand = np.random.rand(4, 4, 4, 2).astype('float32')


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()

    def forward(self, input, grid):
        return F.grid_sample(input, grid, mode='bilinear', padding_mode='reflection', align_corners=True)


def export_onnx_model(onnx_model_file):
    dev = torch.device('cuda:0')

    torch_input = torch.from_numpy(input_rand).half().to(dev) #rand(4, 1, 4, 4) # N C H W
    torch_grid = torch.from_numpy(grid_rand).half().to(dev) #rand(4, 4, 4, 2)

    model = MyModel()
    # print float32 result of this input for trt reference
    # use dynamic_axes to denote the batch dim
    print(model(torch.from_numpy(input_rand[0:2, :, :, :]).float().to(dev), torch.from_numpy(grid_rand[0:2, :, :, :]).float().to(dev)))
    torch.onnx.export( model, (torch_input, torch_grid), onnx_model_file, verbose=True, 
        input_names=['input', 'grid'],output_names=['output'],opset_version = 13,
        custom_opsets={"custom_domain": 1},
        dynamic_axes={"input" : {0: "batch_size"}, "grid" : {0: "batch_size"}})


def modify_onnx(onnx_model_file):
    graph = gs.import_onnx(onnx.load(onnx_model_file))
    assert(graph is not None)

    for node in graph.nodes:
        print("pick graph node: {}".format(node))
        if node.op == 'GridSampler':
            print("find GridSmpler")
            _, c, h, w = node.inputs[0].shape
            _, h_g, w_g, _ = node.inputs[1].shape
            align_corners = node.attrs['align_corners']
            inter_mode = node.attrs['mode']
            pad_mode = node.attrs['padding_mode']
            m_type = 0 if node.inputs[0].dtype == np.float32 else 1
            buffer = np.array([c, h, w, h_g, w_g], dtype=np.int64).tobytes('C') \
              + np.array([inter_mode, pad_mode], dtype=np.int32).tobytes('C') \
              + np.array([align_corners], dtype=np.bool).tobytes('C') \
              + np.array([m_type], dtype=np.int32).tobytes('C')
            node.attrs = {'name':'GridSampler', 'version':'1', 'namespace':"", 'data':buffer}
            node.op = 'TRT_PluginV2'
    
    onnx.save(gs.export_onnx(graph), onnx_model_file)
            

# The Onnx path is used for Onnx models.
def build_engine_onnx(model_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        # builder.max_workspace_size = common.GiB(1)
        # builder.fp16_mode = True
        builder.max_batch_size = 1 # always 1 for explicit batch
        config = builder.create_builder_config()
        # need to be set along with fp16_mode if config is specified.        
        config.set_flag(trt.BuilderFlag.FP16)
        profile = builder.create_optimization_profile()
        profile.set_shape('input', (1, 1, 4, 4), (2, 1, 4, 4), (4, 1, 4, 4))
        profile.set_shape('grid', (1, 4, 4, 2), (2, 4, 4, 2), (4, 4, 4, 2))
        config.add_optimization_profile(profile)
        config.max_workspace_size = common.GiB(1)
        print("trt engine: {}".format(config))

        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(model_file, 'rb') as model:
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None
        return builder.build_engine(network, config)


if __name__=='__main__':

    onnx_model_file = "grid_sample.onnx"
    export_onnx_model(onnx_model_file)
    modify_onnx(onnx_model_file)

    # Build a TensorRT engine.
    with build_engine_onnx(onnx_model_file) as engine:
        # Inference is the same regardless of which parser is used to build the engine, since the model architecture is the same.
        # Allocate buffers and create a CUDA stream.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine, True, 2)
        # Contexts are used to perform inference.
        with engine.create_execution_context() as context:
            # test 1. float16 input, via nvprof, you can see __half populated template function is called
            # test 2. Dims of input and grid is -1 on batch dim. Set context binding shape and feed proper data
            input = input_rand[0:2, :, :, :].astype('float16')
            grid = grid_rand[0:2, :, :, :].astype('float16')
            context.set_binding_shape(0, (2, 1, 4, 4))
            context.set_binding_shape(1, (2, 4, 4, 2))
            inputs[0].host = input
            inputs[1].host = grid
            trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            print("{} output: {}".format(onnx_model_file, trt_outputs))
