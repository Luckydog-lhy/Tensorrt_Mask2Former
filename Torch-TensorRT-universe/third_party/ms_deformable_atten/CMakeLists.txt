# ======================= ms_deformable_atten =======================
add_definitions("-DCUDA_HAS_FP16=1")
add_definitions("-D__CUDA_NO_HALF_OPERATORS__")
add_definitions("-D__CUDA_NO_HALF_CONVERSIONS__")
add_definitions("-D__CUDA_NO_HALF2_OPERATORS__")


include_directories(
        ${PROJECT_SOURCE_DIR}/third_party/ms_deformable_atten/include
)

FILE(GLOB DEFORMABLE_ATTEN
        ${PROJECT_SOURCE_DIR}/third_party/ms_deformable_atten/include/*/*.cu
        )
add_library(deformable_atten SHARED ${DEFORMABLE_ATTEN} )
target_link_libraries(
        deformable_atten ${TORCH_LIBRARIES}
)
