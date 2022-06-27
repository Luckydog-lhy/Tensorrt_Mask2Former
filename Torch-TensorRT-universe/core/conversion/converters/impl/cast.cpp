#include <torch/torch.h>
#include "core/conversion/converters/converter_util.h"
#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"
#include "core/util/trt_util.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto cast_registrations TORCHTRT_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern(
            {"aten::to.dtype(Tensor self, int dtype, bool non_blocking=False, bool copy=False, int? memory_format=None) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               auto output_dtype = args[1].unwrapToScalar().to<int64_t>();
               auto trt_dtype = util::ScalarTypeToTRTDataType(static_cast<at::ScalarType>(output_dtype));
               auto casted_itensor = castITensor(ctx, self, trt_dtype);
               auto output = ctx->AssociateValueAndTensor(n->outputs()[0], casted_itensor);
               LOG_DEBUG("[aten::to.dtype] Output tensor shape: " << output->getDimensions());

               return true;
             }})
        .pattern(
            {"aten::to.other(Tensor self, Tensor other, bool non_blocking=False, bool copy=False, int? memory_format=None) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               nvinfer1::DataType other_dtype = args[1].ITensorOrFreeze(ctx)->getType();
               auto casted_itensor = castITensor(ctx, self, other_dtype);
               auto output = ctx->AssociateValueAndTensor(n->outputs()[0], casted_itensor);
               LOG_DEBUG("[aten::to.other] Output tensor shape: " << output->getDimensions());

               return true;
             }})
        .pattern(
            {"aten::to.prim_Device(Tensor(a) self, Device? device, int? dtype=None, bool non_blocking=False, bool copy=False) -> (Tensor(b|a))",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               if (args[2].isIValue() && !args[2].IValue()->isScalar()) {
                 auto output = ctx->AssociateValueAndTensor(n->outputs()[0], self);
                 LOG_DEBUG("[aten::to.prim_Device] Output tensor shape: " << output->getDimensions());
                 return true;
               }

               auto output_dtype = args[2].unwrapToScalar().to<int64_t>();
               auto trt_dtype = util::ScalarTypeToTRTDataType(static_cast<at::ScalarType>(output_dtype));
               auto casted_itensor = castITensor(ctx, self, trt_dtype);
               auto output = ctx->AssociateValueAndTensor(n->outputs()[0], casted_itensor);
               LOG_DEBUG("[aten::to.prim_Device] Output tensor shape: " << output->getDimensions());

               return true;
             }});

/*
 *               nvinfer1::ITensor* output;
              if (args[0].isITensor()){
                output = ctx->AssociateValueAndTensor(n->outputs()[0], args[0].ITensor());
              } else{
                auto t = args[0].unwrapToTensor();
                auto const_out = tensor_to_const(ctx, t, util::node_info(n).c_str());
                output = ctx->AssociateValueAndTensor(n->outputs()[0], const_out);
              }
              LOG_DEBUG("Output tensor shape: " << output->getDimensions());
 */

auto Int_registrations TORCHTRT_UNUSED =
            RegisterNodeConversionPatterns()
                    .pattern(
                            {"aten::Int.Tensor(Tensor a) -> (int)",
                             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                                 nvinfer1::ITensor* output;
                                 auto trt_dtype = util::ScalarTypeToTRTDataType(at::kInt);
                                 if (args[0].isITensor()) {
                                     auto self = args[0].ITensor();
                                     auto casted_itensor = castITensor(ctx, self, trt_dtype);
                                     output = ctx->AssociateValueAndTensor(n->outputs()[0], casted_itensor);
                                 }else{
                                     auto t = args[0].unwrapToTensor();
                                     auto const_out = tensor_to_const(ctx, t, util::node_info(n).c_str());
                                     auto casted_itensor  = castITensor(ctx, const_out, trt_dtype);
                                     output = ctx->AssociateValueAndTensor(n->outputs()[0], casted_itensor);
                                 }
                                 LOG_DEBUG("[aten::Int] Output tensor shape: " << output->getDimensions());
                                 return true;

                            }});


auto ZerosLike_registrations TORCHTRT_UNUSED =
            RegisterNodeConversionPatterns()
                    .pattern(
                            {"aten::zeros_like(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, int? memory_format=None) -> (Tensor)",
                             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                                 auto self = args[0].ITensorOrFreeze(ctx);
                                 std::vector<int64_t> shape = util::toVec(self->getDimensions());
                                 auto zerosTensor = tensor_to_const(ctx, torch::zeros(shape));

                                 nvinfer1::DataType other_dtype = self->getType();
                                 auto casted_itensor = castITensor(ctx, zerosTensor, other_dtype);
                                 nvinfer1::ITensor* output = ctx->AssociateValueAndTensor(n->outputs()[0], casted_itensor);
                                 LOG_DEBUG("[aten::Int] Output tensor shape: " << output->getDimensions());
                                 return true;
                             }});

//
// clang-format on
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
