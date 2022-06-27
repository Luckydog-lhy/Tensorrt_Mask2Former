//
// Created by hongwei03 on 2022/2/17.
//
#include <torch/script.h> // One-stop header.
#include "torch_tensorrt/torch_tensorrt.h"
#include <iostream>
#include <memory>
using namespace  std;
int seg_encoder_main(int argc, const char* argv[]) {
    torch::jit::Module module;

    auto mod = torch::jit::load("/media/hongwei03/4E69B77F0E1A4357/TensorRT/Mask2Former/output/sem_seg_decoder_torchscrpit.pt",at::Device(at::DeviceType::CUDA, 0));
    std::cout << "ok\n";
    mod.eval();
    mod.to(at::Device(at::DeviceType::CUDA, 0));
    torch::Tensor in3              = torch::ones({ 1, 256, 200, 336},torch::kFloat).to(at::Device(at::DeviceType::CUDA, 0));
    torch::Tensor in2              = torch::ones({ 1, 256, 100, 168},torch::kFloat).to(at::Device(at::DeviceType::CUDA, 0));
    torch::Tensor in1              = torch::ones({ 1, 256, 50, 84},torch::kFloat).to(at::Device(at::DeviceType::CUDA, 0));
    torch::Tensor in0              = torch::ones({ 1, 256, 25, 42},torch::kFloat).to(at::Device(at::DeviceType::CUDA, 0));

    std::vector<torch::IValue> inputs = {in0,in1,in2,in3};
    auto out = mod.forward(inputs );
    for(auto&  output:out.toTuple()->elements()){

        cout<<"torch_trt: "<<output<<endl;// 0.0492  0.0169  0.2507  0.5662  0.0325  0.0511  0.1484  0.0576
    }
    std::vector<int64_t> InputArraysShape0 = in0.sizes().vec();
    std::vector<int64_t> InputArraysShape1 = in1.sizes().vec();
    std::vector<int64_t> InputArraysShape2 = in2.sizes().vec();
    std::vector<int64_t> InputArraysShape3 = in3.sizes().vec();

    std::vector<std::vector<int64_t> > fixed_sizes = {InputArraysShape0,InputArraysShape1,InputArraysShape2,InputArraysShape3};
    auto compile_settings = torch_tensorrt::ts::CompileSpec(fixed_sizes);
    compile_settings.enabled_precisions = {torch::kFloat,torch::kFloat,torch::kFloat,torch::kFloat};

    compile_settings.truncate_long_and_double = true;
    auto trt_mod = torch_tensorrt::torchscript::compile(mod, compile_settings);
    trt_mod.save("/media/hongwei03/4E69B77F0E1A4357/TensorRT/Mask2Former/output/sem_seg_decoder_torchscrpit.pt");
    /*


       0.1523  0.1525  0.1466
  0.1519  0.1521  0.1465
  0.1514  0.1517  0.1464
  0.1510  0.1512  0.1462
  0.1506  0.1508  0.1461
  0.1501  0.1503  0.1460
 -2.4339 -2.4339 -2.4339
 -2.4339 -2.4339 -2.4339
 -2.4339 -2.4339 -2.4339
 -2.4339 -2.4339 -2.4339
 -2.4339 -2.4339 -2.4339
 -2.4339 -2.4339 -2.4339
 -2.4339 -2.4339 -2.4339
 -2.4339 -2.4339 -2.4339
 -2.4339 -2.4339 -2.4339
 -2.4339 -2.4339 -2.4339
 -2.4339 -2.4339 -2.4339
 -2.4339 -2.4339 -2.4339
 -2.4339 -2.4339 -2.4339
 -2.4339 -2.4339 -2.4339
 -2.4339 -2.4339 -2.4339
 -2.4339 -2.4339 -2.4339
[ CUDAFloatType{1,100,200,336} ]
     */
    cout<<" ========================= start trt_test =================== "<<endl;
    auto out_trt = trt_mod.forward(inputs );
    for(auto&  output:out_trt.toTuple()->elements()){

        cout<<"torch_trt: "<<output<<endl;// 0.0492  0.0169  0.2507  0.5662  0.0325  0.0511  0.1484  0.0576
    }

    return 0;
}



int main(int argc, const char* argv[]) {
    torch::jit::Module module;

    auto mod = torch::jit::load("/media/hongwei03/4E69B77F0E1A4357/TensorRT/Mask2Former/output/sem_seg_encoder_torchscrpit.pt",at::Device(at::DeviceType::CUDA, 0));
    std::cout << "ok\n";
    mod.eval();
    mod.to(at::Device(at::DeviceType::CUDA, 0));

    auto dtype_input = torch::kFloat16;
    torch::Tensor in0              = torch::ones({ 1, 256, 200, 336},torch::kFloat).to(at::Device(at::DeviceType::CUDA, 0));
    torch::Tensor in1              = torch::ones({ 1, 512, 100, 168},torch::kFloat).to(at::Device(at::DeviceType::CUDA, 0));
    torch::Tensor in2              = torch::ones({ 1, 1024, 50, 84},torch::kFloat).to(at::Device(at::DeviceType::CUDA, 0));
    torch::Tensor in3              = torch::ones({ 1, 2048, 25, 42},torch::kFloat).to(at::Device(at::DeviceType::CUDA, 0));
    torch::Tensor mask_flatten     = torch::ones({ 1, 22050},torch::kFloat).to(at::Device(at::DeviceType::CUDA, 0));
    torch::Tensor reference_points = torch::ones({ 1, 22050,3,2},torch::kFloat).to(at::Device(at::DeviceType::CUDA, 0));


    std::vector<torch::IValue> inputs = {in0,in1,in2,in3,mask_flatten,reference_points};
    auto out = mod.forward(inputs );
//    for(auto&  output:out.toTuple()->elements()){
//
//        cout<<"torch_ori: "<<output<<endl;//0.0003  0.0469  0.1922  0.0126  0.1595  0.0140  0.0085  0.0203
//    }


    //-5.0678 -342.5218 -408.2026
    std::vector<int64_t> InputArraysShape0 = in0.sizes().vec();
    std::vector<int64_t> InputArraysShape1 = in1.sizes().vec();
    std::vector<int64_t> InputArraysShape2 = in2.sizes().vec();
    std::vector<int64_t> InputArraysShape3 = in3.sizes().vec();

    std::vector<int64_t> mask_flattenShape2 = mask_flatten.sizes().vec();
    std::vector<int64_t> reference_pointsShape2 = reference_points.sizes().vec();


    std::vector<std::vector<int64_t> > fixed_sizes = {InputArraysShape0,InputArraysShape1,InputArraysShape2,InputArraysShape3,mask_flattenShape2,reference_pointsShape2};
    auto compile_settings = torch_tensorrt::ts::CompileSpec(fixed_sizes);
    compile_settings.enabled_precisions = {torch::kFloat,torch::kFloat,torch::kFloat,torch::kFloat,torch::kFloat,torch::kFloat};

    compile_settings.truncate_long_and_double = true;
    auto trt_mod = torch_tensorrt::torchscript::compile(mod, compile_settings);
    trt_mod.save("/media/hongwei03/4E69B77F0E1A4357/TensorRT/Mask2Former/output/sem_seg_encoder_torchscrpit.pt");
    /*


       0.1523  0.1525  0.1466
  0.1519  0.1521  0.1465
  0.1514  0.1517  0.1464
  0.1510  0.1512  0.1462
  0.1506  0.1508  0.1461
  0.1501  0.1503  0.1460
  0.1496  0.1498  0.1458
  0.1491  0.1493  0.1457
  0.1486  0.1488  0.1456
  0.1481  0.1483  0.1454
  0.1476  0.1478  0.1453
  0.1471  0.1473  0.1451
  0.1466  0.1468  0.1450
  0.1461  0.1464  0.1449
  0.1457  0.1459  0.1448
  0.1453  0.1455  0.1447
  0.1449  0.1451  0.1446
  0.1446  0.1448  0.1446
  0.0716  0.0717  0.1523
[ CUDAFloatType{1,256,200,336} ]
     */
    cout<<" ========================= start trt_test =================== "<<endl;
    auto out_trt = trt_mod.forward(inputs );
    for(auto&  output:out_trt.toTuple()->elements()){

        cout<<"torch_trt: "<<output<<endl;// 0.0492  0.0169  0.2507  0.5662  0.0325  0.0511  0.1484  0.0576
    }

    return 0;
}

int encoder_main(int argc, const char* argv[]) {
    torch::jit::Module module;

    auto mod = torch::jit::load("/media/hongwei03/4E69B77F0E1A4357/TensorRT/wenet/exp/20210601_u2++_conformer_exp/encoder.pt",at::Device(at::DeviceType::CUDA, 0));
    std::cout << "ok\n";
    mod.eval();
    mod.to(at::Device(at::DeviceType::CUDA, 0));
    torch::Tensor in0 = torch::ones({ 32,100,80},torch::kFloat).to(at::Device(at::DeviceType::CUDA, 0));
    torch::Tensor in1 = torch::ones({ 32},torch::kFloat).to(at::Device(at::DeviceType::CUDA, 0));
    auto out = mod.forward({in0,in1} );
    cout<<"torch_ori: "<<out<<endl;
    //-5.0678 -342.5218 -408.2026
    std::vector<int64_t> InputArraysShape0 = in0.sizes().vec();
    std::vector<int64_t> InputArraysShape1 = in1.sizes().vec();
    std::vector<std::vector<int64_t> > fixed_sizes = {InputArraysShape0,InputArraysShape1};
    auto compile_settings = torch_tensorrt::ts::CompileSpec(fixed_sizes);
    compile_settings.enabled_precisions = {torch::kFloat};

    compile_settings.truncate_long_and_double = true;
    auto trt_mod = torch_tensorrt::torchscript::compile(mod, compile_settings);
    trt_mod.save("/home/hongwei03/netease/VideoEngine/xgboost_realtime_predict_test/models/face_net/face_ir_100_test.pt");
    auto out_trt = trt_mod.forward({in0,in1} );
    cout<<"torch_trt: "<<out_trt<<endl;
    return 0;
}



int export_bert(){

    auto mod = torch::jit::load("/home/hongwei03/下载/bert.torchscript.pt",at::Device(at::DeviceType::CUDA, 0));
    std::cout << "ok\n";
    mod.eval();
    mod.to(at::Device(at::DeviceType::CUDA, 0));
    torch::Tensor in0 = torch::ones({ 12,64},torch::kLong).to(at::Device(at::DeviceType::CUDA, 0));
    torch::Tensor in1 = torch::ones({ 1},torch::kLong).to(at::Device(at::DeviceType::CUDA, 0));
    torch::Tensor in2 = torch::ones({ 12,64},torch::kLong).to(at::Device(at::DeviceType::CUDA, 0));
    auto out = mod.forward({in0,in1,in2} );
    cout<<"torch_ori: "<<out<<endl;
    /*
     *
     */
    //Columns 501 to 510-0.3414 -0.4345  0.7932  0.0657 -0.3011  0.4331 -0.8478  0.8211 -0.9559  1.0576
//    mod.eval();
    std::vector<int64_t> InputArraysShape0 = in0.sizes().vec();
    std::vector<int64_t> InputArraysShape1 = in1.sizes().vec();
    std::vector<int64_t> InputArraysShape2 = in2.sizes().vec();
    std::vector<std::vector<int64_t> > fixed_sizes = {InputArraysShape0,InputArraysShape1,InputArraysShape2};


    auto compile_settings = torch_tensorrt::ts::CompileSpec(fixed_sizes);
    compile_settings.enabled_precisions = {torch::kFloat};

    compile_settings.truncate_long_and_double = true;
    auto trt_mod = torch_tensorrt::torchscript::compile(mod, compile_settings);
    trt_mod.save("/home/hongwei03/下载/bert.torchscript_3.pt");


    in0 = torch::ones({ 12,64},torch::kFloat).to(at::Device(at::DeviceType::CUDA, 0));
    in1 = torch::ones({ 1},torch::kFloat).to(at::Device(at::DeviceType::CUDA, 0));
    in2 = torch::ones({ 12,64},torch::kFloat).to(at::Device(at::DeviceType::CUDA, 0));

    auto out_trt = trt_mod.forward({in0,in1,in2} );
    cout<<"torch_trt: "<<out_trt<<endl;
    return 0;
}
