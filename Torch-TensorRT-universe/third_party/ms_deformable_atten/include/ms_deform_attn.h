/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

/*!
* Copyright (c) Facebook, Inc. and its affiliates.
* Modified by Bowen Cheng from https://github.com/fundamentalvision/Deformable-DETR
*/

#pragma once
#define WITH_CUDA
#include "cpu/ms_deform_attn_cpu.h"
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/torch.h>
#ifdef WITH_CUDA
#include "cuda/ms_deform_attn_cuda.h"
#endif


at::Tensor
ms_deform_attn_forward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step)
{
    if (value.type().is_cuda())
    {
#ifdef WITH_CUDA
        return ms_deform_attn_cuda_forward(
            value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor>
ms_deform_attn_backward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int im2col_step)
{
    if (value.type().is_cuda())
    {
#ifdef WITH_CUDA
        return ms_deform_attn_cuda_backward(
            value, spatial_shapes, level_start_index, sampling_loc, attn_weight, grad_output, im2col_step);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}




at::Tensor ms_deform_attn_tensorrt_forward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight
    )
{
    if (value.type().is_cuda())
    {

#ifdef WITH_CUDA
        return ms_deform_attn_cuda_forward(
            value, spatial_shapes, level_start_index, sampling_loc, attn_weight, 128);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

//TORCH_LIBRARY(ms_deform_attn, m) {
//  m.def("ms_deform_attn_forward", ms_deform_attn_tensorrt_forward);
//}

//TORCH_LIBRARY_IMPL(tensorrt, CUDA, m) {
//  m.impl("ms_deform_attn_forward", ms_deform_attn_tensorrt_forward);
//}

