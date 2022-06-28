/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 
#include <vector>
#include <string>
#include <NvInfer.h>
#include <iostream>
// +------- Debug wrapper --------------------------------------------------------------------------
#if DEBUG
#define WHERE_AM_I() do {printf("[%s]: this=->%p\n",__func__,this);} while(0);
#else
#define WHERE_AM_I()
#endif // DEBUG



namespace torch_tensorrt {
namespace core {
namespace plugins {
namespace impl {


// +------- Plguin ---------------------------------------------------------------------------------
namespace
{
    static const char* PLUGIN_NAME{"LayerNormGB"};
    static const char* PLUGIN_VERSION{"1"};
} // namespace


// +------- Plugin body ----------------------------------------------------------------------------
class LayerNormGBPlugin: public nvinfer1::IPluginV2DynamicExt
{
    private:
    std::string name_;
    std::string namespace_="torch_tensorrt";

    public:
    LayerNormGBPlugin(const std::string& name) : name_(name)
    {
        WHERE_AM_I();
    }

    LayerNormGBPlugin(const std::string& name, const void* data, size_t length) : name_(name)
    {
        WHERE_AM_I();
    }

    LayerNormGBPlugin() = delete;
    ~LayerNormGBPlugin()
    {
        WHERE_AM_I();
    }
    size_t getSerializationSize() const noexcept override
    {
        WHERE_AM_I();
        return 0;
    }
    void serialize(void *buffer) const noexcept override
    {
        WHERE_AM_I();
    }
    IPluginV2DynamicExt* clone() const noexcept override
    {
        WHERE_AM_I();
        return new LayerNormGBPlugin(name_);
    }
    int getNbOutputs() const noexcept override
    {
        WHERE_AM_I();
        return 1;
    }

    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override
    {
        WHERE_AM_I();
        return inputs[0];
    }

    bool supportsFormatCombination(int32_t pos, const nvinfer1::PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        WHERE_AM_I();
        if (inOut[pos].format != nvinfer1::TensorFormat::kLINEAR)
        {
            return false;
        }

        bool res = false;
        switch (pos)
        {
            case 0:
//                res = (inOut[pos].type == DataType::kFLOAT);
                res = (inOut[pos].type == nvinfer1::DataType::kHALF or inOut[pos].type == nvinfer1::DataType::kFLOAT );break;
//                std::cout<<"res: "<<res<<std::endl;

            case 1:
                res = inOut[pos].type == inOut[0].type; break;
            case 2:
                 res = inOut[pos].type == inOut[0].type; break;
            case 3:
                 res = inOut[pos].type == inOut[0].type; break;
            case 5:
                 res = inOut[pos].type == inOut[0].type; break;
            default: // should NOT be here
            break;
        }
        return res;

    }

    nvinfer1::DataType getOutputDataType(int outputIndex, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override
    {
        WHERE_AM_I();
        return nvinfer1::DataType::kFLOAT;
    }

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs,const nvinfer1::DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override
    {
        WHERE_AM_I();
    }

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs, const nvinfer1::PluginTensorDesc* outputs,int32_t nbOutputs) const noexcept override
    {
        WHERE_AM_I();
        return 0;
    }

    void setPluginNamespace(const char* szNamespace) noexcept override
    {
        WHERE_AM_I();
        namespace_ = szNamespace;
    }
    const char* getPluginNamespace() const noexcept override
    {
        WHERE_AM_I();
        return namespace_.c_str();
    }
    const char* getPluginType() const noexcept override
    {
        WHERE_AM_I();
        return PLUGIN_NAME;
    }
    const char* getPluginVersion() const noexcept override
    {
        WHERE_AM_I();
        return PLUGIN_VERSION;
    }
    int initialize() noexcept override
    {
        WHERE_AM_I();
        return 0;
    }
    void terminate() noexcept override
    {
        WHERE_AM_I();
        return;
    }

    void destroy() noexcept override
    {
        WHERE_AM_I();
    }

    int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
}; // class LayerNormGBPlugin

class LayerNormGBPluginCreator : public nvinfer1::IPluginCreator
{
    private:
    static nvinfer1::PluginFieldCollection fc_;
    static std::vector<nvinfer1::PluginField> attr_;
    std::string namespace_="torch_tensorrt";

    public:
    LayerNormGBPluginCreator()
    {
        fc_.nbFields = attr_.size();
        fc_.fields = attr_.data();
    }

    ~LayerNormGBPluginCreator() {}

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override
    {
        WHERE_AM_I();
        return new LayerNormGBPlugin(name);
    }

    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override
    {
        return new LayerNormGBPlugin(name, serialData, serialLength);
    }
    void setPluginNamespace(const char* szNamespace) noexcept override
    {
        namespace_ = szNamespace;
    }
    const char* getPluginNamespace() const noexcept override
    {
        return namespace_.c_str();
    }
    const char* getPluginName() const noexcept override
    {
        return PLUGIN_NAME;
    }
    const char* getPluginVersion() const noexcept override
    {
        return PLUGIN_VERSION;
    }
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override
    {
        return &fc_;
    }
}; // class LayerNormGBPluginCreator





} // namespace impl
} // namespace plugins
} // namespace core
} // namespace torch_tensorrt

