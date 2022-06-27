mkdir build
cd build
cmake -DCUDA_CUDA_LIBRARY=true -DPYTHON_LIBRARY=/workspace/wenet/lib/libpython3.7m.so -DPYTHON_INCLUDE_DIR=/workspace/wenet/include/python3.7m -DPYTHON_EXECUTABLE=/workspace/wenet/bin/python  ..
make -j6
