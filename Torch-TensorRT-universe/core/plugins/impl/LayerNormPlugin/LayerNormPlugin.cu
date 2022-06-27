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

#include "core/plugins/impl/LayerNormPlugin/LayerNormPlugin.h"

#include "core/plugins/plugins.h"
#include <numeric>
#include <stdexcept>

using namespace nvinfer1;



namespace torch_tensorrt {
namespace core {
namespace plugins {
namespace impl {


PluginFieldCollection LayerNormGBPluginCreator::fc_{};
std::vector<PluginField> LayerNormGBPluginCreator::attr_;


//=============================== layernorm fp32 =================================

#define FINAL_MASK 0xffffffff
template<typename T>
__inline__ __device__ T warpReduceSum(T val)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
    return val;
}


template<typename T>
__inline__ __device__ T blockReduceSum(T val)
{
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = warpReduceSum<T>(val);

    if (lane == 0)
        shared[wid] = val;

    __syncthreads();

    // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
    // blockDim.x is not divided by 32
    val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : (T)(0.0f);
    val = warpReduceSum<T>(val);

    return val;
}

template<typename T>
inline __device__ T ldg(const T* val) {
    return __ldg(val);
}

template<typename T>
__global__ void generalLayerNorm(
        const T* __restrict input, const T* __restrict gamma, const T* __restrict beta, T* output, int m, int n)
{

    const int tid = threadIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        local_sum += (float)(ldg(&input[blockIdx.x * n + i]));
    }

    mean = blockReduceSum(local_sum);

    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float local_var_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float diff = (float)(ldg(&input[blockIdx.x * n + i])) - s_mean;
        local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + 1e-6f);
    }
    __syncthreads();


    for (int i = tid; i < n; i += blockDim.x) {
        float beta_val = (beta == nullptr) ? 0.0f : (float)ldg(&beta[i]);
        output[blockIdx.x * n + i] =
                (T)((((float)input[blockIdx.x * n + i] - s_mean) * s_variance) * (float)(ldg(&gamma[i])) + beta_val);
    }
}



//============================== layernorm fp16 ===================================

#include <cuda_fp16.h>

static const float HALF_FLT_MAX = 65504.F;


template<typename T>
struct TypeConverter {using Type = half2;}; // keep for generality

template<>
struct TypeConverter<half2> {using Type = half;};

template<>
struct TypeConverter<half> {using Type = half2;};

template<typename T, int NUM>
__inline__ __device__ T warpReduceSumV2(T* val)
{
#pragma unroll
    for (int i = 0; i < NUM; i++) {
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            val[i] += __shfl_xor_sync(FINAL_MASK, val[i], mask, 32);
    }
    return (T)(0.0f);
}



template<typename T>
__device__ __forceinline__ T clamp_inf_for_half(const float input)
{
    if (std::is_same<T, half>::value == true) {
        // clamp inf values to enable fp16 training
        return (float)input > 0.0f ? min(input, HALF_FLT_MAX - 1000) : max(input, -HALF_FLT_MAX + 1000);
    }
    else {
        return input;
    }
}


// Convert float to type2 (applied to half2 and bfloat162)
template<typename T>
inline __device__ T float2type2(float a);

template<>
inline __device__ half2 float2type2(float a) {
    return __float2half2_rn(a);
}

#ifdef ENABLE_BF16
template<>
inline __device__ __nv_bfloat162 float2type2(float a) {
    return __float2bfloat162_rn(a);
}
#endif // ENABLE_BF16

// Convert float to type (applied to half and bfloat16)
template<typename T>
inline __device__ T float2type(float a);

template<>
inline __device__ half float2type(float a) {
    return __float2half_rn(a);
}

#ifdef ENABLE_BF16
template<>
inline __device__ __nv_bfloat16 float2type(float a) {
    return __float2bfloat16_rn(a);
}
#endif // ENABLE_BF16



template<typename T, int NUM>
__inline__ __device__ T blockReduceSumV2(T* val)
{
    static __shared__ T shared[NUM][33];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    warpReduceSumV2<T, NUM>(val);

    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < NUM; i++) {
            shared[i][wid] = val[i];
        }
    }

    __syncthreads();

    bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
    for (int i = 0; i < NUM; i++) {
        val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
    }
    warpReduceSumV2<T, NUM>(val);
    return (T)0.0f;
}




#ifdef ENABLE_BF16
template<>
inline __device__ __nv_bfloat162 hadd2(__nv_bfloat162 a, __nv_bfloat162 b) {
    return bf16hadd2(a, b);
}
#endif // ENABLE_BF16

template<typename T>
inline __device__ T hsub2(T a, T b) {
    return __hsub2(a, b);
}

#ifdef ENABLE_BF16
template<>
inline __device__ __nv_bfloat162 hsub2(__nv_bfloat162 a, __nv_bfloat162 b) {
    return bf16hsub2(a, b);
}
#endif // ENABLE_BF16

template<typename T>
inline __device__ T hmul2(T a, T b) {
    return __hmul2(a, b);
}

template<typename T>
inline __device__ T hadd2(T a, T b) {
    return __hadd2(a, b);
}

#ifdef ENABLE_BF16
template<>
inline __device__ __nv_bfloat162 hadd2(__nv_bfloat162 a, __nv_bfloat162 b) {
    return bf16hadd2(a, b);
}
#endif // ENABLE_BF16



template<typename T,int UNROLL_FACTOR>
__global__ void generalAddBiasResidualLayerNormOpt2(T* normed_output,
                                                    T* output,
                                                    const T* __restrict residual,
                                                    const T* __restrict gamma,
                                                    const T* __restrict beta,
                                                    int m,
                                                    int n)
{
    __shared__ float s_mean;
    __shared__ float s_variance;
    float x_sum = 0.0f;
    float x2_sum = 0.0f;
    const int b_offset = blockIdx.x * n;
    using T1 = typename TypeConverter<T>::Type;

#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = b_offset + i;
        float val_1 = 0.0f;
        float val_2 = 0.0f;
        T tmp;
        tmp = ldg(&residual[index]);
        val_1 += static_cast<float>(tmp.x);
        val_2 += static_cast<float>(tmp.y);
        tmp.x = float2type<T1>(val_1);
        tmp.y = float2type<T1>(val_2);
        x_sum += val_1 + val_2;
        output[index] = tmp;
        x2_sum += val_1 * val_1 + val_2 * val_2;
    }
    float sums[2];
    sums[0] = x_sum;
    sums[1] = x2_sum;
    blockReduceSumV2<float, 2>(sums);

    if (threadIdx.x == 0) {
        s_mean = sums[0] / n / 2;
        s_variance = rsqrtf(sums[1] / n / 2 - s_mean * s_mean + 1e-6f);
    }
    __syncthreads();

    T mean_2 = float2type2<T>(s_mean);
    T var_2 = float2type2<T>(s_variance);



#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = b_offset + i;
        T val = hmul2(hmul2(hsub2(output[index], mean_2), var_2), ldg(&gamma[i]));
        val = hadd2(val, ldg(&beta[i]));
        normed_output[index] = val;

    }
}


#define HALF_LAYERNORM_OPT2(UNROLL_FACTOR)                                                                             \
    generalAddBiasResidualLayerNormOpt2<T2,UNROLL_FACTOR><<<grid, block, 0, stream>>>(      \
        (T2*)out, (T2*)out, (const T2*)input,(const T2*)gamma,(const T2*)beta, m, half_n);
// bool IS_OUTPUT, bool IS_BIAS, bool IS_RESIDUAL, bool IS_BETA



#include <iostream>
int32_t LayerNormGBPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const void* input =   inputs[0];
    const void* gamma =   inputs[1];
    const void* beta  =   inputs[2];
    void* out         =   outputs[0];
//================= faster transformer =================//
    int m = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
    int n = inputDesc[0].dims.d[2];
    int k = inputDesc[0].dims.d[1];
    dim3 grid(m);

    if(inputDesc[0].type==DataType::kHALF )
    {
        int half_n = n / 2;
        int half_n_32 = (half_n + 31) / 32 * 32;
        dim3 block(min(half_n_32, 512));
        int rolls_per_thread = half_n / block.x;
        int unroll_factor = 8;
        while (unroll_factor > rolls_per_thread && unroll_factor > 1) {
            unroll_factor /= 2;
        }
//        std::cout<<"unroll_factor: "<<unroll_factor<<std::endl;
        using T2 = typename TypeConverter<half>::Type;
//        HALF_LAYERNORM_OPT2(8);
        if (unroll_factor == 1) {
            HALF_LAYERNORM_OPT2(1);
        }
        else if (unroll_factor == 2) {
            HALF_LAYERNORM_OPT2(2);
        }
        else if (unroll_factor == 3) {
            HALF_LAYERNORM_OPT2(3);
        }
        else if (unroll_factor == 4) {
            HALF_LAYERNORM_OPT2(4);
        }
        else if (unroll_factor == 8) {
            HALF_LAYERNORM_OPT2(8);
        }

    }
    else{
        dim3 block(min(n, 1024));
//        std::cout<<"float32: "<<std::endl;
                        /* For general cases, n is equal to hidden_units, e.g., 512/1024.
                            Since we have warp shuffle inside the code, block.x % 32 should be 0.
                        */
        if (n % 32 != 0) {
            block.x = 1024;
        }
        generalLayerNorm<float><<<grid, block, 0, stream>>>(
                                (const float *)input,(const float *)gamma,(const float *)beta,(float *)out , m, n);  // For gpt-3


    }
    return 0;
}
REGISTER_TORCHTRT_PLUGIN(LayerNormGBPluginCreator);

} // namespace impl
} // namespace plugins
} // namespace core
} // namespace torch_tensorrt