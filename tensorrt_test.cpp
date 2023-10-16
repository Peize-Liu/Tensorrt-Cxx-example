#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <NvInfer.h>
#include <memory.h>
#include <fstream>
#include <vector>
#include <NvOnnxParser.h>
#include <stdint.h>
#include "common/buffers.h"
#include <stdint.h>
// #define UM


//user should realize
const std::vector<std::string> kInputnames = {"input"};
const std::vector<std::string> kOutputnames = {"reference_output_disparity"};

class Logger: public nvinfer1::ILogger{
  void log(Severity severity, const char* msg) noexcept override{
    if(severity <= Severity::kWARNING){
      std::cout << msg <<std::endl;
    }
  }
};

//A tensor Excutor is a wrapper of context and stream
class TensorExcutor{
public:
  TensorExcutor(std::shared_ptr<nvinfer1::ICudaEngine> engine){
    context_ = engine->createExecutionContext();
    if (!context_){
      printf("[Tensor basic] Create context failed\n");
    }
    cudaStreamCreate(&stream_);
    if (cudaGetLastError() != cudaSuccess){
      printf("[Tensor basic] Create stream failed\n");
    }
    
    buffer_manager_ = new samplesCommon::BufferManager(engine, 1, context_);
    //depends on your engine so should realize in child class
    this->input_dims_ = engine->getBindingDimensions(0);
    this->output_dims_ = engine->getBindingDimensions(1);
    printf("Engine input data type:%d\n", engine->getBindingDataType(0));
    printf("Engine output data type:%d\n", engine->getBindingDataType(1)); //float32_t
    memset(buffer_manager_->getHostBuffer(kOutputnames[0]), 0, output_dims_.d[1] * output_dims_.d[2] * output_dims_.d[3] * sizeof(float));
    //memset intput buffer
    memset(buffer_manager_->getHostBuffer(kInputnames[0]), 0, input_dims_.d[1] * input_dims_.d[2] * input_dims_.d[3] * sizeof(float));

  }
  ~TensorExcutor(){
    // delete buffer_manager_;
    // cudaStreamDestroy(stream_);
    // context_->destroy();
  }
  int32_t setInputData(cv::Mat & left_image, cv::Mat & right_image){
    if (left_image.empty() || right_image.empty()){
      // printf("input image empty\n");
      return -1;
    }
    cv::Mat left_image_mono, right_image_mono, input_image;
    if (left_image.channels() != 1 || right_image.channels() != 1){
      // printf("input image channel not 1\n");
      cv::cvtColor(left_image, left_image_mono, cv::COLOR_BGR2GRAY);
      cv::cvtColor(right_image, right_image_mono, cv::COLOR_BGR2GRAY);
    } else {
      left_image_mono = left_image;
      right_image_mono = right_image;
    }
    if (left_image_mono.rows != input_dims_.d[2] || left_image_mono.cols != input_dims_.d[3]){
      // printf("input image size not match\n");
      cv::resize(left_image_mono, left_image_mono, cv::Size(input_dims_.d[3], input_dims_.d[2]));
      cv::resize(right_image_mono, right_image_mono, cv::Size(input_dims_.d[3], input_dims_.d[2]));
    }
    cv::vconcat(left_image_mono, right_image_mono, input_image);
    // cv::imshow("input", input_image);
    input_image.convertTo(input_image_, CV_32FC1, 1.0/255.0);//
    memcpy(buffer_manager_->getHostBuffer(kInputnames[0]), input_image_.data, input_dims_.d[1] * input_dims_.d[2] * input_dims_.d[3] * sizeof(float));
    // buffer_manager_->copyInputToDevice();
    #ifndef UM
    buffer_manager_->copyInputToDeviceAsync(stream_);
    #endif

    return 0; 
  }

  int32_t doInfference(){
    bool status = this->context_->enqueueV2(buffer_manager_->getDeviceBindings().data(), stream_, nullptr);
    // bool status = this->context_->executeV2(buffer_manager_->getDeviceBindings().data());
    if (!status){
      printf("do inference failed\n");
      return -1;
    }
    return 0;
  }

  int32_t getOutputData(){
    #ifndef UM
    buffer_manager_->copyOutputToHostAsync(stream_);
    #endif
    cudaStreamSynchronize(stream_);
    // buffer_manager_->copyOutputToHost();
    float* host_data = static_cast<float*>(buffer_manager_->getHostBuffer(kOutputnames[0]));
    cv::Mat output_mat(output_dims_.d[1], output_dims_.d[2], CV_32FC1, host_data);
    cv::normalize(output_mat, output_mat, 0, 255, cv::NORM_MINMAX,CV_8UC1);
    cv::applyColorMap(output_mat, output_mat, cv::COLORMAP_JET);
    // cv::imshow("output", output_mat);
    // cv::waitKey(0);
    return 0;
  }

private:
  samplesCommon::BufferManager* buffer_manager_;
  nvinfer1::IExecutionContext* context_;
  cudaStream_t stream_;

  //realize in child class
  bool dim_recongized_ = false;
  nvinfer1::Dims input_dims_;
  nvinfer1::Dims output_dims_;
  cv::Mat input_image_;
  cv::Mat output_image_;
};

int main(int argc, char** argv){
  std::string trt_engine_path = argv[1];
  if (trt_engine_path.empty()){
    std::cout << "Please specify the path to the engine file." << std::endl;
    return -1;
  }
  Logger logger;
  std::cout << "Engine path: " << trt_engine_path << std::endl;
  //creat builder
  #if 0
  std::string onnx_path = argv[1];
  nvinfer1::IBuilder* nv_builder = nvinfer1::createInferBuilder(logger);
  if (!nv_builder){
    printf("[Tensor basic] Builder created failed\n");
    return -1;
  }
  //construct network
  const uint32_t explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  nvinfer1::INetworkDefinition* nv_network_ptr = nv_builder->createNetworkV2(explicit_batch);
  //build engine from onnx //TODO: 
  nvinfer1::IBuilder* nv_builder_ptr = nvinfer1::createInferBuilder(logger);
  //create build config
  nvinfer1::IBuilderConfig* nv_config_ptr = nv_builder_ptr->createBuilderConfig();
  //create parser
  nvonnxparser::IParser* nv_parser_ptr = nvonnxparser::createParser(*nv_network_ptr, logger);
  //parse engine
  auto parsed = nv_parser_ptr->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
  if (!parsed){
    printf("[Tensor basic] Parser failed\n");
    return -1;
  }
  //enable fp16
  nv_config_ptr->setFlag(nvinfer1::BuilderFlag::kFP16);

  //stream for profiling
  cudaStream_t profile_stream;
  cudaError_t cuda_err = cudaStreamCreate(&profile_stream);
  if (cuda_err != cudaSuccess){
    printf("[Tensor basic] Create stream failed\n");
    return -1;
  }
  nv_config_ptr->setProfileStream(profile_stream);
  nvinfer1::IHostMemory* plan = nv_builder_ptr->buildSerializedNetwork(*nv_network_ptr, *nv_config_ptr);
  if (!plan){
    printf("[Tensor basic] Build engine failed\n");
    return -1;
  }
  auto runtime = nvinfer1::createInferRuntime(logger);
  nvinfer1::ICudaEngine* nv_engine_ptr = runtime->deserializeCudaEngine(plan->data(), plan->size(), nullptr);
  if (!nv_engine_ptr){
    printf("[Tensor basic] Engine deserrialized failed\n");
    return -1;
  }
  nv_network_ptr->getNbInputs();
  nv_network_ptr->getNbOutputs();
  printf("Input number: %d\n", nv_network_ptr->getNbInputs());
  printf("Output number: %d\n", nv_network_ptr->getNbOutputs());
  printf("engine builded\n");
  if (!nv_parser_ptr){
    printf("[Tensor basic] Parser created failed\n");
    return -1;
  }
  //write engine
  std::ofstream engine_file("./test_engine.trt", std::ios::binary);
  auto serialize_engine = nv_engine_ptr->serialize();
  engine_file.write(static_cast<const char*>(serialize_engine->data()), serialize_engine->size());
  engine_file.close();
  printf("engine writed\n");
  nv_engine_ptr->destroy();
  #else
  //create runtime
  nvinfer1::IRuntime* nv_runtime = nvinfer1::createInferRuntime(logger);
  // init engine
  std::ifstream engine_file(trt_engine_path, std::ios::binary);
  if (!engine_file){
    printf("[Tensor basic] Engine file deserrialized failed\n");
    return -2;
  }
  std::stringstream engine_buffer;
  engine_buffer << engine_file.rdbuf();
  std::string plan = engine_buffer.str();
  printf("engine size: %d\n", plan.size());
  auto nv_engine_ptr =  nv_runtime->deserializeCudaEngine(plan.data(), plan.size(), nullptr);
  std::shared_ptr<nvinfer1::ICudaEngine> nv_engine_ptr2(nv_engine_ptr, samplesCommon::InferDeleter());
  if (!nv_engine_ptr){
    printf("[Tensor basic] Engine deserrialized failed\n");
    return -3;
  }
  TensorExcutor tensor_excutor(nv_engine_ptr2);
  printf("engine init\n");
  cv::Mat l_image = cv::imread("/root/workspace/left.png");
  cv::Mat r_image = cv::imread("/root/workspace/right.png");
  if  (l_image.empty() || r_image.empty()){
    printf("read image failed\n");
    return -1;
  }


  std::vector<TensorExcutor> excutor_lists;
  for (int i = 0; i < 4; i++){
    TensorExcutor tensor_excutor(nv_engine_ptr2);
    excutor_lists.push_back(tensor_excutor);
  }

  for (auto && iter : excutor_lists){
    iter.setInputData(l_image, r_image);
    printf("input data set\n");
  }


  time_t start, end;
  start = clock();
  for(int i = 0; i < 1000; i++){
    for (auto && iter : excutor_lists){
      iter.setInputData(l_image, r_image);
      // printf("do inference\n");
    }
    for (auto && iter : excutor_lists){
      iter.doInfference();
      // printf("do inference\n");
    }
    for (auto && iter : excutor_lists){
      iter.getOutputData();
      // printf("get output\n");
    }
  }
  end = clock();

  printf("time: %f\n", (double)(end - start)/CLOCKS_PER_SEC);



  tensor_excutor.setInputData(l_image, r_image);
  printf("input data set\n");
  tensor_excutor.doInfference();
  tensor_excutor.getOutputData();
#endif
  return 0;
}

