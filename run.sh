#! /bin/bash
# set -ex

# pip install pycuda
# pip install onnxruntime-gpu
# pip install onnx-graphsurgeon
reset
LD_PRELOAD=TensorRT/build/out/libnvinfer_plugin.so python test_plugin_result.py
