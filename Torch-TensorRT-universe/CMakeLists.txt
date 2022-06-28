cmake_minimum_required(VERSION 3.15)
project(Torch_trt LANGUAGES CXX CUDA)



set(CMAKE_INSTALL_PREFIX  ${PROJECT_SOURCE_DIR})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)




#
############################################################################################
# CUDA targets
#find_package(CUDA 10.2 REQUIRED)
if (DEFINED GPU_ARCHS)
    message(STATUS "GPU_ARCHS defined as ${GPU_ARCHS}. Generating CUDA code for SM ${GPU_ARCHS}")
    separate_arguments(GPU_ARCHS)
else()
    list(APPEND GPU_ARCHS
#            35
#            53
            61
            70
            86
            )

    string(REGEX MATCH "aarch64" IS_ARM "${TRT_PLATFORM_ID}")
    if (IS_ARM)
        # Xavier (SM72) only supported for aarch64.
        list(APPEND GPU_ARCHS 72)
    endif()

    if (CUDA_VERSION VERSION_GREATER_EQUAL 11.0)
        # Ampere GPU (SM80) support is only available in CUDA versions > 11.0
        list(APPEND GPU_ARCHS 80)
    else()
        message(WARNING "Detected CUDA version is < 11.0. SM80 not supported.")
    endif()

    message(STATUS "GPU_ARCHS is not defined. Generating CUDA code for default SMs: ${GPU_ARCHS}")
endif()
set(BERT_GENCODES)
# Generate SASS for each architecture
foreach(arch ${GPU_ARCHS})
    if (${arch} GREATER_EQUAL 70)
        set(BERT_GENCODES "${BERT_GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
    endif()
    set(GENCODES "${GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()
# Generate PTX for the last architecture in the list.
list(GET GPU_ARCHS -1 LATEST_SM)
set(GENCODES "${GENCODES} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")
set(BERT_GENCODES "${BERT_GENCODES} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")
if (${LATEST_SM} GREATER_EQUAL 70)
    set(BERT_GENCODES "${BERT_GENCODES} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")
endif()


#
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
                        -gencode=arch=compute_61,code=\\\"sm_70,compute_61\\\" \
                        -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
                        -gencode=arch=compute_86,code=\\\"sm_86,compute_86\\\" \
                        ")
##                      -rdc=true")
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
set(CMAKE_CUDA_ARCHITECTURES 61 70 86 )
#message("-- Assign GPU architecture (sm=70,75)")
#set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
#set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
# set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall  --ptxas-options=-v --resource-usage")
#set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall")
##
#set(CMAKE_CXX_STANDARD "${CXX_STD}")
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++${CXX_STD}")
#
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
## set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3 --ptxas-options=--verbose")
#set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3")





#
#find_package(CUDA 10.2 REQUIRED)
#if(${CUDA_VERSION_MAJOR} VERSION_GREATER_EQUAL "11")
#    add_definitions("-DENABLE_BF16")
#    message("CUDA_VERSION ${CUDA_VERSION_MAJOR} is greater or equal than 11, enable -DENABLE_BF16 flag")
#endif()
#
#option(BUILD_TF "Build in TensorFlow mode" OFF)
#option(BUILD_PYT "Build in PyTorch TorchScript class mode" OFF)
#option(BUILD_TRT "Build projects about TensorRT" OFF)
#if(NOT BUILD_MULTI_GPU)
#    option(BUILD_MULTI_GPU "Build project about multi-GPU" OFF)
#endif()
#if(NOT USE_TRITONSERVER_DATATYPE)
#    option(USE_TRITONSERVER_DATATYPE "Build triton backend for triton server" OFF)
#endif()
#
#option(SPARSITY_SUPPORT "Build project with Ampere sparsity feature support" OFF)
#
#if(BUILD_MULTI_GPU)
#    message(STATUS "Add DBUILD_MULTI_GPU, requires MPI and NCCL")
#    add_definitions("-DBUILD_MULTI_GPU")
#    set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
#    find_package(MPI REQUIRED)
#    find_package(NCCL REQUIRED)
#    #if(${NCCL_VERSION} LESS 2.7)
#    #  message(FATAL_ERROR "NCCL_VERSION ${NCCL_VERSION} is less than 2.7")
#    #endif()
#    set(CMAKE_MODULE_PATH "") # prevent the bugs for pytorch building
#endif()
#
#if(BUILD_PYT)
#    if(DEFINED ENV{NVIDIA_PYTORCH_VERSION})
#        if($ENV{NVIDIA_PYTORCH_VERSION} VERSION_LESS "20.03")
#            message(FATAL_ERROR "NVIDIA PyTorch image is too old for TorchScript mode.")
#        endif()
#        if($ENV{NVIDIA_PYTORCH_VERSION} VERSION_EQUAL "20.03")
#            add_definitions(-DLEGACY_THS=1)
#        endif()
#    endif()
#endif()
#
#if(USE_TRITONSERVER_DATATYPE)
#    message("-- USE_TRITONSERVER_DATATYPE")
#    add_definitions("-DUSE_TRITONSERVER_DATATYPE")
#endif()
#
#set(CXX_STD "14" CACHE STRING "C++ standard")
#
#set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
#
#set(TF_PATH "" CACHE STRING "TensorFlow path")
#set(CUSPARSELT_PATH "" CACHE STRING "cuSPARSELt path")
#
#
#
#list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)
#
## profiling
#option(USE_NVTX "Whether or not to use nvtx" OFF)
#if(USE_NVTX)
#    message(STATUS "NVTX is enabled.")
#    add_definitions("-DUSE_NVTX")
#endif()
#
#
#
## setting compiler flags
#set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")
#set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall -ldl")
#
#set(SM_SETS 52 60 61 70 75)
#set(USING_WMMA False)
#set(FIND_SM False)
#
#
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
#                        -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
#                        -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
#                        ")
##                      -rdc=true")
#set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
#set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
#set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
#message("-- Assign GPU architecture (sm=70,75)")
#
#
#
#set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
#set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
## set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall  --ptxas-options=-v --resource-usage")
#set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall")
#
#set(CMAKE_CXX_STANDARD "${CXX_STD}")
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++${CXX_STD}")
#
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
## set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3 --ptxas-options=--verbose")
#set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3")
#
#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
#
#set(COMMON_HEADER_DIRS
#        ${PROJECT_SOURCE_DIR}
#        ${CUDA_PATH}/include
#        )
#
#set(COMMON_LIB_DIRS
#        ${CUDA_PATH}/lib64
#        )
#
#
#list(APPEND COMMON_HEADER_DIRS ${MPI_INCLUDE_PATH})
##
##if(USE_TRITONSERVER_DATATYPE)
##    list(APPEND COMMON_HEADER_DIRS ${PROJECT_SOURCE_DIR}/../repo-core-src/include)
##endif()
#
#include_directories(
#        ${COMMON_HEADER_DIRS}
#)






## =========================================== openmp opencv ========================================================
message(STATUS "===========openmp opencv============")
find_package(OpenMP)
if(OpenMP_CXX_FOUND OR OPENMP_FOUND)
    message(STATUS "===========OpenMP module============")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP_INCLUDE_DIRS: ${OpenMP_INCLUDE_DIRS}")
    message(STATUS "OpenMP_LIBS: ${OpenMP_LIBS}")
    message(STATUS "OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP_C_FLAGS: ${OpenMP_C_FLAGS}")
endif()

list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/3rd/opencv3")
message(STATUS "===========opencv module============")
find_package( OpenCV  REQUIRED   )
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "opencv Not Found!")
else()
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
    message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
endif (NOT OpenCV_FOUND)

message(STATUS "===========torch module============")
#find_package(PythonInterp REQUIRED)
list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/3rd/libtorch")
find_package(Torch REQUIRED)
message(STATUS "===========  TORCH_LIBRARIES  ============")
message(${TORCH_LIBRARIES})

# ==================== Tensorrt Lib ==================================
include_directories(
        ${PROJECT_SOURCE_DIR}/3rd
	    ${PROJECT_SOURCE_DIR}/3rd/Trt/include
)
link_directories(
        ${PROJECT_SOURCE_DIR}/3rd/Trt/lib
)

list(APPEND Trt_LIBRARY
        nvinfer_plugin
        nvinfer
        nvonnxparser
        )

# =====================================  third =========================================
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/ms_deformable_atten)

#include_directories(
#        ${PROJECT_SOURCE_DIR}/third_party/cub
#)

# ======================  torch_tensorrt include ============================
include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/core
        ${PROJECT_SOURCE_DIR}/core/conversion
        ${PROJECT_SOURCE_DIR}/core/ir
        ${PROJECT_SOURCE_DIR}/core/lowering
        ${PROJECT_SOURCE_DIR}/core/partitioning
        ${PROJECT_SOURCE_DIR}/core/plugins
        ${PROJECT_SOURCE_DIR}/core/plugins/impl/common
        ${PROJECT_SOURCE_DIR}/core/plugins/impl/common/kernels
        ${PROJECT_SOURCE_DIR}/core/runtime
        ${PROJECT_SOURCE_DIR}/core/util
        ${PROJECT_SOURCE_DIR}/third_party/ms_deformable_atten/include
)

# ======================  torch_tensorrt plugin ============================

FILE(GLOB TORCHTRT_PLUGIN
        ${PROJECT_SOURCE_DIR}/core/plugins/*.cpp
        ${PROJECT_SOURCE_DIR}/core/plugins/impl/interpolate_plugin.cpp
#        ${PROJECT_SOURCE_DIR}/core/plugins/impl/common/*.cpp
        )
FILE(GLOB TORCHTRT_PLUGIN_KERNEL
#        ${PROJECT_SOURCE_DIR}/core/plugins/impl/common/kernels/*.cu
        ${PROJECT_SOURCE_DIR}/core/plugins/impl/groupNormalizationPlugin/*.cu
        ${PROJECT_SOURCE_DIR}/core/plugins/impl/deformAttnPlugin/*.cu
        ${PROJECT_SOURCE_DIR}/core/plugins/impl/LayerNormPlugin/*.cu
        ${PROJECT_SOURCE_DIR}/core/plugins/impl/multiscaleDeformableAttnPlugin/*.cu
        )

# ======================  torch_tensorrt core ============================
FILE(GLOB TORCHTRT
        ${PROJECT_SOURCE_DIR}/core/*.cpp
        ${PROJECT_SOURCE_DIR}/core/lowering/*.cpp
        ${PROJECT_SOURCE_DIR}/core/lowering/passes/*.cpp
        ${PROJECT_SOURCE_DIR}/core/ir/*.cpp
        ${PROJECT_SOURCE_DIR}/core/partitioning/*.cpp
        ${PROJECT_SOURCE_DIR}/core/runtime/*.cpp
        )
FILE(GLOB TORCHTRT_CONVERSION
        ${PROJECT_SOURCE_DIR}/core/conversion/*.cpp
        ${PROJECT_SOURCE_DIR}/core/conversion/conversionctx/*.cpp
        ${PROJECT_SOURCE_DIR}/core/conversion/converters/*.cpp
        ${PROJECT_SOURCE_DIR}/core/conversion/converters/impl/*.cpp
        ${PROJECT_SOURCE_DIR}/core/conversion/evaluators/*.cpp
        ${PROJECT_SOURCE_DIR}/core/conversion/tensorcontainer/*.cpp
        ${PROJECT_SOURCE_DIR}/core/conversion/var/*.cpp
)
FILE(GLOB TORCHTRT_UTIL
        ${PROJECT_SOURCE_DIR}/core/util/*.cpp
        ${PROJECT_SOURCE_DIR}/core/util/logging/*.cpp
        )

list(APPEND TORCHTRT  ${TORCHTRT_CONVERSION} ${TORCHTRT_UTIL} ${TORCHTRT_PLUGIN} ${TORCHTRT_PLUGIN_KERNEL})
add_library(torchTrt SHARED ${TORCHTRT} )
target_link_libraries(
        torchTrt
        deformable_atten
        ${Trt_LIBRARY}
        ${TORCH_LIBRARIES}
#        -lcudart -lcublas -lcublasLt -lnvinfer -lcublasMMWrapper
)



# ======================  torch_tensorrt cpp import ============================

include_directories(${PROJECT_SOURCE_DIR}/cpp/include)
FILE(GLOB CPPSRC
        ${PROJECT_SOURCE_DIR}/cpp/src/*.cpp
        )
add_library(torchTrtCppImport SHARED ${CPPSRC})
target_link_libraries(
        torchTrtCppImport
        torchTrt
)


# ====================  Test EXE =======================

add_executable(torch_trt_runtime_test ${PROJECT_SOURCE_DIR}/examples/torchtrt_runtime_example/main.cpp)
target_link_libraries(
        torch_trt_runtime_test
        torchTrtCppImport
)


add_executable(torchTrtTest ${PROJECT_SOURCE_DIR}/examples/torch_trt_compile/TorchTrt_test.cpp)
target_link_libraries(
        torchTrtTest
        torchTrtCppImport
)



# ======================  torch_tensorrt python ============================
include_directories(/root/anaconda3/envs/wenet/include/python3.7m)
link_directories(/root/anaconda3/envs/wenet/lib)
#list(APPEND CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/3rd/pybind11)
#find_package( pybind11  REQUIRED   )

include_directories(
        ${PROJECT_SOURCE_DIR}/py/csrc
        )
FILE(GLOB TORCHTRT_PYTHON
        ${PROJECT_SOURCE_DIR}/py/torch_tensorrt/csrc/*.cpp
        )
list(APPEND CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/3rd/pybind11)
find_package( pybind11  REQUIRED   )
message("${pybind11_VERSION}")
message("${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
#add_definitions("-fPIC")
pybind11_add_module(torchTrtPythonImport   ${TORCHTRT_PYTHON})

target_link_libraries(
        torchTrtPythonImport
        PRIVATE torchTrt
)

#
#add_library(torchTrtPythonImport SHARED ${TORCHTRT_PYTHON})
#target_link_libraries(
#        torchTrtPythonImport
#        torchTrtCppImport
#        torchTrt
#)
