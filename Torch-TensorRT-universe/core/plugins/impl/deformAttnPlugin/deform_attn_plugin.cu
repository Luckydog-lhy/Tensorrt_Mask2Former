#include "core/plugins/impl/deformAttnPlugin/deform_attn_plugin.h"
#include "third_party/ms_deformable_atten/include/cuda/ms_deform_attn_cuda.h"
#include "NvInferPlugin.h"
#include "NvInferPluginUtils.h"
#include "core/plugins/plugins.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace plugins {
namespace impl {

/*
 * deformAttnPlugin class implementations
 */

deformAttnPlugin::deformAttnPlugin(int32_t im2col_step)
    : im2col_step_(im2col_step) {}
    deformAttnPlugin::deformAttnPlugin(const char* data, size_t length) {
  std::istringstream data_stream(std::string(data, length));

  torch::serialize::InputArchive input_archive;
  input_archive.load_from(data_stream);
  {
    torch::IValue value;
    input_archive.read("im2col_step", value);
    im2col_step_ = (int32_t)value.toInt();
  }

}

int deformAttnPlugin::getNbOutputs() const noexcept {
  return 1;
}

const char* deformAttnPlugin::getPluginType() const noexcept {
  return "deformAttnPlugin";
}

const char* deformAttnPlugin::getPluginVersion() const noexcept {
  return "1";
}

const char* deformAttnPlugin::getPluginNamespace() const noexcept {
  return "torch_tensorrt";
}

nvinfer1::IPluginV2DynamicExt* deformAttnPlugin::clone() const noexcept {
  return new deformAttnPlugin(im2col_step_);
}

nvinfer1::DimsExprs deformAttnPlugin::getOutputDimensions(
    int outputIndex,
    const nvinfer1::DimsExprs* inputs,
    int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
  nvinfer1::DimsExprs output;
  output.nbDims = inputs[0].nbDims - 1;
  output.d[0] = exprBuilder.constant(inputs[0].d[0]->getConstantValue());
  output.d[1] = exprBuilder.constant(inputs[0].d[1]->getConstantValue());
  output.d[2] = exprBuilder.constant(inputs[0].d[2]->getConstantValue() * inputs[0].d[3]->getConstantValue());
  return output;
}

nvinfer1::DataType deformAttnPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs)
    const noexcept {
  return nvinfer1::DataType::kFLOAT;
}

int deformAttnPlugin::initialize() noexcept {
  return 0;
}

void deformAttnPlugin::serialize(void* buffer) const noexcept {
  std::string data = serializeToString();
  size_t size = getSerializationSize();
  data.copy((char*)buffer, size);
}

std::string deformAttnPlugin::serializeToString() const noexcept {
  torch::serialize::OutputArchive output_archive;
  output_archive.write("im2col_step", torch::IValue((int64_t)im2col_step_));
  std::ostringstream data_str;
  output_archive.save_to(data_str);

  return data_str.str();
}

size_t deformAttnPlugin::getSerializationSize() const noexcept {
  return serializeToString().size();
}

bool deformAttnPlugin::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc* inOut,
    int nbInputs,
    int nbOutputs) noexcept {
//  if (pos < 0 || pos > 1) {
//    LOG_ERROR("There should be exactly 2 connections to the plugin - 1 input, 1 output");
//  }
//  if (nbInputs != 1) {
//    LOG_ERROR("Expected a single tensor as input to normalize plugin");
//  }
//  if (nbOutputs != 1) {
//    LOG_ERROR("Expected a single tensor as output to normalize plugin");
//  }
//
//  const nvinfer1::PluginTensorDesc& in = inOut[0];
//
//  if (pos == 0) {
//    return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == nvinfer1::TensorFormat::kLINEAR);
//  }
//
//  // pos == 1, accessing information about output tensor
//  const nvinfer1::PluginTensorDesc& out = inOut[1];
//
//  return (in.type == out.type) && (in.format == out.format);


    if (inOut[pos].format != nvinfer1::TensorFormat::kLINEAR)
    {
        return false;
    }


//    if (pos == 0) {
//      const nvinfer1::PluginTensorDesc& in = inOut[0];
//      return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == nvinfer1::TensorFormat::kLINEAR);
//    }
    bool res = false;
    switch (pos)
    {
        case 0:
//                res = (inOut[pos].type == DataType::kFLOAT) || (inOut[pos].type == DataType::kHALF);
            res = (inOut[pos].type == nvinfer1::DataType::kFLOAT);
//                std::cout<<"res: "<<res<<std::endl;
            break;
        case 1:
            res = inOut[pos].type == nvinfer1::DataType::kINT32;
            break;
        case 2:
            res = inOut[pos].type == nvinfer1::DataType::kINT32;
            break;
        case 3:
            res = inOut[pos].type == inOut[0].type;
            break;
        case 4:
            res = inOut[pos].type == inOut[0].type;
            break;
        case 5:
            res = inOut[pos].type == inOut[0].type;
            break;
        default: // should NOT be here
            break;
    }

    return res;


}

void deformAttnPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in,
    int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int nbOutputs) noexcept {
  dtype_ = nvinfer1::DataType::kFLOAT;
}

size_t deformAttnPlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs,
    int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int nbOutputs) const noexcept {
  return 0;
}

int deformAttnPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {


    auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA).requires_grad(false);
    auto options2 = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(torch::kCUDA).requires_grad(false);
    at::Tensor value             = torch::from_blob((void*)inputs[0], util::toVec(inputDesc[0].dims), options).to(torch::kFloat);
    at::Tensor spatial_shapes    = torch::from_blob((void*)inputs[1], util::toVec(inputDesc[1].dims), options2).to(torch::kLong);
//    std::cout<<spatial_shapes<<std::endl;
    at::Tensor level_start_index = torch::from_blob((void*)inputs[2], util::toVec(inputDesc[2].dims), options2).to(torch::kLong);
    at::Tensor sampling_loc      = torch::from_blob((void*)inputs[3], util::toVec(inputDesc[3].dims), options).to(torch::kFloat);
    at::Tensor attn_weight       = torch::from_blob((void*)inputs[4], util::toVec(inputDesc[4].dims), options).to(torch::kFloat);
    at::Tensor output            = torch::from_blob((void*)outputs[0], util::toVec(outputDesc[0].dims), options).to(torch::kFloat);
//  at::Tensor value =
//          at::from_blob((void*)inputs[0], util::toVec(inputDesc[0].dims), [](void*) {}, {at::kCUDA}).to(torch::kFloat);
//  at::Tensor spatial_shapes =
//          at::from_blob((void*)inputs[1], util::toVec(inputDesc[1].dims), [](void*) {}, {at::kCUDA}).to(torch::kLong);
//  at::Tensor level_start_index =
//          at::from_blob((void*)inputs[2], util::toVec(inputDesc[2].dims), [](void*) {}, {at::kCUDA}).to(torch::kLong);
//  at::Tensor sampling_loc =
//          at::from_blob((void*)inputs[3], util::toVec(inputDesc[3].dims), [](void*) {}, {at::kCUDA}).to(torch::kFloat);
//
//  at::Tensor attn_weight =
//          at::from_blob((void*)inputs[4], util::toVec(inputDesc[4].dims), [](void*) {}, {at::kCUDA}).to(torch::kFloat);
//  at::Tensor output =
//          at::from_blob(outputs[0], util::toVec(outputDesc[0].dims), [](void*) {}, {at::kCUDA}).to(torch::kFloat);

//  at::cuda::CUDAStream torch_stream = at::cuda::getStreamFromPool();
//  at::cuda::CUDAStreamGuard torch_guard(torch_stream);
//
//  cudaEvent_t event;
//  cudaEventCreate(&event);
//  cudaEventRecord(event, stream);
//  cudaStreamWaitEvent(torch_stream.stream(), event, 0);
  {
      at::Tensor result = ms_deform_attn_cuda_forward(value, spatial_shapes, level_start_index, sampling_loc,
                                                        attn_weight, im2col_step_);
      output.copy_(result);
  }
//  cudaEvent_t torch_event;
//  cudaEventCreate(&torch_event);
//  cudaEventRecord(torch_event, torch_stream.stream());
//  cudaStreamWaitEvent(stream, torch_event, 0);
//
//  cudaEventDestroy(event);
//  cudaEventDestroy(torch_event);
  return 0;
}

/*
 * deformAttnPluginCreator class implementations
 */
deformAttnPluginCreator::deformAttnPluginCreator() {
  mPluginAttributes.emplace_back(nvinfer1::PluginField("im2col_step", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* deformAttnPluginCreator::getPluginNamespace() const noexcept {
  return "torch_tensorrt";
}

const char* deformAttnPluginCreator::getPluginName() const noexcept {
  return "deformAttnPlugin";
}

const char* deformAttnPluginCreator::getPluginVersion() const noexcept {
  return "1";
}

nvinfer1::IPluginV2* deformAttnPluginCreator::createPlugin(
    const char* name,
    const nvinfer1::PluginFieldCollection* fc) noexcept {
  int32_t im2col_step = 128;
  std::vector<int32_t> axes;
  int32_t keep_dims = 0;
  for (int i = 0; i < fc->nbFields; i++) {
    std::string field_name(fc->fields[i].name);
    if (field_name.compare("im2col_step") == 0) {
        im2col_step = *static_cast<const int32_t*>(fc->fields[i].data);
    }
  }
  deformAttnPlugin* plugin = new deformAttnPlugin(im2col_step);
  return plugin;
}

nvinfer1::IPluginV2* deformAttnPluginCreator::deserializePlugin(
    const char* name,
    const void* serialData,
    size_t serialLength) noexcept {
  name_ = name;
  auto plugin = new deformAttnPlugin((const char*)serialData, serialLength);
  return plugin;
}

const nvinfer1::PluginFieldCollection* deformAttnPluginCreator::getFieldNames() noexcept {
  return nullptr;
}

REGISTER_TORCHTRT_PLUGIN(deformAttnPluginCreator);

} // namespace impl
} // namespace plugins
} // namespace core
} // namespace torch_tensorrt
