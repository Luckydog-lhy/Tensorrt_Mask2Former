//
// Created by hongwei03 on 2022/6/14.
//

#include "NvInfer.h"
#include "core/conversion/converters/converters.h"
#include "core/conversion/tensorcontainer/TensorContainer.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

#include <ATen/ATen.h>
#include <vector>

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {
void deform_attn(ConversionCtx* ctx,const torch::jit::Node* n,args& args){
    LOG_WARNING("group_normalize run in tensorrt plugin");
    nvinfer1::ITensor* value             = args[0].ITensor();
    nvinfer1::ITensor* spatial_shapes    = args[1].ITensor();
    nvinfer1::ITensor* level_start_index = args[2].ITensor();
    nvinfer1::ITensor* sampling_loc      = args[3].ITensor();
    nvinfer1::ITensor* attn_weight       = args[4].ITensor();
//    nvinfer1::ITensor* attn_weight = tensor_to_const(ctx,  args[4].unwrapToTensor());

    std::vector<nvinfer1::PluginField> f;
    int im2col_step = 128;
    f.emplace_back(nvinfer1::PluginField("im2col_step", &im2col_step, nvinfer1::PluginFieldType::kINT32, 1));
    nvinfer1::PluginFieldCollection fc;
    fc.nbFields = f.size();
    fc.fields = f.data();


    auto creator = getPluginRegistry()->getPluginCreator("deformAttnPlugin", "1", "torch_tensorrt");
    auto plugin = creator->createPlugin("deformAttnPluginTorchTRT", &fc);
    std::vector<nvinfer1::ITensor*> inputs = {value,spatial_shapes,level_start_index,sampling_loc,attn_weight};
    auto deform_attn_layer = ctx->net->addPluginV2(inputs.data(),5, *plugin);
    deform_attn_layer->setName(util::node_info(n).c_str());
    auto out_tensor = deform_attn_layer->getOutput(0);
    auto layer_output = ctx->AssociateValueAndTensor(n->outputs()[0], out_tensor );
    TORCHTRT_CHECK(deform_attn_layer, "Unable to create group normalization plugin from node" << *n);
    LOG_DEBUG("deformAttn layer output tensor shape: " << layer_output->getDimensions());
}


void multi_deform_attn(ConversionCtx* ctx,const torch::jit::Node* n,args& args){
    LOG_WARNING("group_normalize run in tensorrt plugin");
    nvinfer1::ITensor* value             = args[0].ITensor();
    nvinfer1::ITensor* spatial_shapes    = args[1].ITensor();
    nvinfer1::ITensor* level_start_index = args[2].ITensor();
    nvinfer1::ITensor* sampling_loc      = args[3].ITensor();
    nvinfer1::ITensor* attn_weight       = args[4].ITensor();
//   nvinfer1::ITensor* attn_weight = tensor_to_const(ctx,  args[4].unwrapToTensor());
    std::vector<nvinfer1::PluginField> f;
    int im2col_step = 128;
    f.emplace_back(nvinfer1::PluginField("im2col_step", &im2col_step, nvinfer1::PluginFieldType::kINT32, 1));
    nvinfer1::PluginFieldCollection fc;
    fc.nbFields = f.size();
    fc.fields = f.data();
    auto creator = getPluginRegistry()->getPluginCreator("MultiscaleDeformableAttnPlugin_TRT", "1", "torch_tensorrt");
    auto plugin = creator->createPlugin("multiDeformAttnPluginTorchTRT", &fc);
    std::vector<nvinfer1::ITensor*> inputs = {value,spatial_shapes,level_start_index,sampling_loc,attn_weight};
    auto deform_attn_layer = ctx->net->addPluginV2(inputs.data(),5, *plugin);
    deform_attn_layer->setName(util::node_info(n).c_str());
    auto deform_attn_out_tensor = deform_attn_layer->getOutput(0);
//    auto layer_output = ctx->AssociateValueAndTensor(n->outputs()[0], out_tensor );
//    TORCHTRT_CHECK(deform_attn_layer, "Unable to create group normalization plugin from node" << *n);
//    LOG_DEBUG("deformAttn layer output tensor shape: " << layer_output->getDimensions());


    auto  deform_attn_out_tensor_dim = deform_attn_out_tensor->getDimensions();
    std::vector<int64_t> new_shape =
            {deform_attn_out_tensor_dim.d[0],deform_attn_out_tensor_dim.d[1],deform_attn_out_tensor_dim.d[2]*deform_attn_out_tensor_dim.d[3]};
    auto shuffle = ctx->net->addShuffle(*deform_attn_out_tensor);
    TORCHTRT_CHECK(shuffle, "Unable to create shuffle layer from node: " << *n);
    shuffle->setReshapeDimensions(util::toDims(new_shape));
    shuffle->setName((util::node_info(n)+"_reshape") .c_str());
    auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], shuffle->getOutput(0));
    LOG_DEBUG("deformAttn layer output tensor shape: " << out_tensor->getDimensions());
}

auto ms_deform_attn_tensorrt_forward_registrations TORCHTRT_UNUSED = RegisterNodeConversionPatterns().pattern(
        {"tensorrt::ms_deform_attn_tensorrt_forward(Tensor _0, Tensor _1, Tensor _2, Tensor _3, Tensor _4) -> (Tensor _0)",
         [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {auto self = args[0].ITensorOrFreeze(ctx);
             multi_deform_attn(ctx,n,args);
             return true;
        }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt