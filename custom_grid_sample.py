#!/usr/bin/python
# -*- coding:utf-8 -*-
import typing

try:
    from torch.onnx import register_custom_op_symbolic
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "This module is only useful in combination with PyTorch. "
        "To install PyTorch see https://pytorch.org/.")
import torch.onnx.symbolic_helper as sym_help
import torch.onnx.symbolic_registry as sym_registry
import torch

_OPSET_VERSION = 1
_registered_ops: typing.AbstractSet[str] = set()

def _reg(symbolic_fn: typing.Callable):
    name = "::%s" % symbolic_fn.__name__
    register_custom_op_symbolic(name, symbolic_fn, _OPSET_VERSION)
    print("register_custom_op_symbolic, name: {}, symbolic_fn: {}, _OPSET_VERSION: {}".format(name, symbolic_fn, _OPSET_VERSION))
    _registered_ops.add(name)

def grid_sampler(g:torch._C.Graph, input:torch._C.Value, grid:torch._C.Value, mode:int, padding_mode:int, align_corners:bool):
    # mode
    #   'bilinear'      : onnx::Constant[value={0}]
    #   'nearest'       : onnx::Constant[value={1}]
    #   'bicubic'       : onnx::Constant[value={2}]
    # padding_mode
    #   'zeros'         : onnx::Constant[value={0}]
    #   'border'        : onnx::Constant[value={1}]
    #   'reflection'    : onnx::Constant[value={2}]
    mode = sym_help._maybe_get_const(mode, "i")
    padding_mode = sym_help._maybe_get_const(padding_mode, "i")
    mode_str = ['bilinear', 'nearest', 'bicubic'][mode]
    padding_mode_str = ['zeros', 'border', 'reflection'][padding_mode]
    align_corners = int(sym_help._maybe_get_const(align_corners, "b"))

    # From opset v13 onward, the output shape can be specified with
    # (N, C, H, W) (N, H_out, W_out, 2) => (N, C, H_out, W_out)
    # g.op(...).setType(input.type().with_sizes(output_shape))
    # print("[grid_sampler] input graph: {}".format(g))
    input_shape = input.type().sizes()
    grid_shape = grid.type().sizes()
    input_strides = input.type().strides()
    grid_strides = grid.type().strides()
    print("[grid_sampler] input_strides: {}, grid_strides: {}".format(input_strides, grid_strides))
    print("[grid_sampler] input_shape: {}, grid_shape: {}".format(input_shape, grid_shape))
    
    if input_shape!=None and grid_shape!=None:
        output_shape = input_shape[:2] + grid_shape[1:3]
        print("[grid_sampler] input_shape -> output_shape: {}".format(output_shape))
    elif input_strides!=None and grid_strides!=None: 
        C = int(input_strides[0]/input_strides[1])
        W_out = int(grid_strides[1]/grid_strides[2])
        H_out = int(grid_strides[0]/grid_strides[1])
        output_shape = [1, C, H_out, W_out]
        print("[grid_sampler] input_strides -> output_shape: {}".format(output_shape))
    else:
        output_shape = [1, 1, 1, 1]
        print("[grid_sampler] nowhere -> output_shape: {}".format(output_shape))

    return g.op("custom_domain::GridSampler", input, grid,
                mode_i=mode,
                padding_mode_i=padding_mode,
                align_corners_i=align_corners).setType(input.type().with_sizes(output_shape))

_reg(grid_sampler)

@torch.jit.script
def custom_grid_sample(input:torch.Tensor, grid:torch.Tensor):
    x = torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    ### RuntimeError: Unsupported: ONNX export of transpose for tensor of unknown rank.
    return x

def test():
    x = torch.rand(size=[1, 256, 112, 192], dtype=torch.float)
    grid = torch.rand(size=[1, 224, 384, 2], dtype=torch.float)
    y = custom_grid_sample(x, grid)
    print("custom_grid_sample: {}".format(y.shape))
    print("custom_grid_sample: {}".format(custom_grid_sample.graph))

    torch.onnx.export(
        model=custom_grid_sample, 
        args=(x, grid),
        f="custom_grid_sample.onnx",
        custom_opsets={"custom_domain": 1},
        opset_version=13, verbose=True, do_constant_folding=True)

if __name__ == '__main__':
    test()
