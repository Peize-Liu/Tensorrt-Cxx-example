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
    this->input_dims_ = engine->getBindingDimensions(0);
    this->output_dims_ = engine->getBindingDimensions(1);
    printf("Engine input data type:%d\n", engine->getBindingDataType(0));
    printf("Engine output data type:%d\n", engine->getBindingDataType(1)); //float32_t
    cudaError_t status =  cudaMallocManaged(&this->unified_input_buffer_, input_dims_.d[1] * input_dims_.d[2] * input_dims_.d[3] * sizeof(float), cudaMemAttachHost);
    if (status != cudaSuccess){
      printf("cudaMallocManaged failed\n");
    }
    status = cudaMallocManaged(&this->unified_output_buffer_, output_dims_.d[1] * output_dims_.d[2] * output_dims_.d[3] * sizeof(float));
    if (status != cudaSuccess){
      printf("cudaMallocManaged failed\n");
    }
    status = cudaStreamAttachMemAsync(this->stream_, this->unified_input_buffer_, 0, cudaMemAttachGlobal);
    if (status != cudaSuccess){
      printf("cudaStreamAttachMemAsync failed\n");
    }

    status = cudaStreamAttachMemAsync(this->stream_, this->unified_output_buffer_, 0, cudaMemAttachHost);
    if (status != cudaSuccess){
      printf("cudaStreamAttachMemAsync failed\n");
    }

  }
  ~TensorExcutor(){
    if (this->unified_input_buffer_== nullptr){
      cudaFree(this->unified_input_buffer_);
    }
    if (this->unified_output_buffer_ == nullptr){
      cudaFree(this->unified_output_buffer_);
    }
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
    input_image.convertTo(input_image_, CV_32FC1, 1.0/255.0);//
    memcpy(this->unified_input_buffer_, input_image_.data, input_dims_.d[1] * input_dims_.d[2] * input_dims_.d[3] * sizeof(float));
    // cudaError_t status = cudaStreamAttachMemAsync(this->stream_, this->unified_input_buffer_, 0, cudaMemAttachGlobal);
    // if (status != cudaSuccess){
    //   printf("cudaStreamAttachMemAsync failed\n");
    // }
    // buffer_manager_->copyInputToDeviceAsync(stream_);
    return 0;
  }

  int32_t doInfference(){
    // bool status = this->context_->enqueueV3(stream_);//not work
    std::vector<void*> buffers;
    buffers.push_back(this->unified_input_buffer_);
    buffers.push_back(this->unified_output_buffer_);
    bool status = this->context_->enqueueV2(buffers.data(), stream_, nullptr);
    // bool status = this->context_->executeV2(buffer_manager_->getDeviceBindings().data());
    if (!status){
      printf("do inference failed\n");
      return -1;
    }
    return 0;
  }

  int32_t getOutputData(){
    // cudaError_t status = cudaStreamAttachMemAsync(this->stream_, this->unified_output_buffer_, 0, cudaMemAttachHost);
    // if (status != cudaSuccess){
    //   printf("cudaStreamAttachMemAsync failed\n");
    // }
    // printf("Debug\n");
    cudaStreamSynchronize(this->stream_);
    // buffer_manager_->copyOutputToHost();
    float* host_data = static_cast<float*>(this->unified_output_buffer_);
    if (this->unified_output_buffer_ == nullptr){
      printf("output null\n");
      return -1;
    }
    cv::Mat output_mat(output_dims_.d[1], output_dims_.d[2], CV_32FC1, host_data);
    cv::normalize(output_mat, output_mat, 0, 255, cv::NORM_MINMAX,CV_8UC1);
    cv::applyColorMap(output_mat, output_mat, cv::COLORMAP_JET);
    // cv::imshow("output", output_mat);
    // cv::waitKey(0);
    return 0;
  }

private:
  void* unified_input_buffer_;
  void* unified_output_buffer_;
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
  // std::shared_ptr<nvinfer1::ICudaEngine> nv_engine_ptr2(nv_engine_ptr, samplesCommon::InferDeleter());
  std::shared_ptr<nvinfer1::ICudaEngine> nv_engine_ptr2(nv_engine_ptr, InferDeleter());
  if (!nv_engine_ptr){
    printf("[Tensor basic] Engine deserrialized failed\n");
    return -3;
  }
  TensorExcutor tensor_excutor(nv_engine_ptr2);
  printf("engine init\n");
  cv::Mat l_image = cv::imread("/home/dji/workspace/Tensorrt-Cxx-example/left.png");
  cv::Mat r_image = cv::imread("/home/dji/workspace/Tensorrt-Cxx-example/right.png");
  if  (l_image.empty() || r_image.empty()){
    printf("read image failed\n");
    return -1;
  }
  tensor_excutor.setInputData(l_image, r_image);
  printf("input data set\n");
  tensor_excutor.doInfference();
  printf("finish infference\n");
  tensor_excutor.getOutputData();

    std::vector<TensorExcutor> excutor_lists;
  for (int i = 0; i < 4; i++){
    TensorExcutor tensor_excutor(nv_engine_ptr2);
    excutor_lists.push_back(tensor_excutor);
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
  return 0;
}

