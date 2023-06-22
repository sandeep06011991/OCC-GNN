#pragma once
/*
* Using thrust device_vectors is too heavy
* Resizes also call fill function to set values to 0
* reverting to using cub
* Required features.
*/
#include "cuda_utils.h"
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <string>
#include <vector>
#include <iterator>
#include <torch/torch.h>
#include <algorithm>
namespace cuslicer{


// inline torch::Tensor getTensor(cuslicer::device_vector<long> &v, c10::TensorOptions opts){
//     if(v.size() == 0){
//       return torch::empty(v.size(),opts);
//     }else{
//       return torch::from_blob(v.ptr(), {(long)v.size()}, opts).clone();

//     }
// auto opts = torch::TensorOptions().dtype(torch::kInt64)\
//     .device(torch::kCUDA, local_gpu_id);

  // This class is completely unnecessary.
  // Its because I used my own allocator.
  template <typename DATATYPE>
  class device_vector{

    size_t allocated = 0;

    size_t free_size= 0;

    static long TOTAL_USED;

    static int DEVICE;

  public:
    //  Todo: Should not be exposed 
    torch::Tensor data;
    torch::Tensor EMPTY;
    
    size_t current_size= 0;

    float getAllocatedMemory(){
      // return (cuda_memory::TOTAL_ALLOCATED  * 1.0) / (1032 * 1032 * 1032);
    }
    
    static void printMemoryStats(){
      // std::cout << "Allocated:";
      // std::cout << (cuda_memory::TOTAL_ALLOCATED  * 1.0) / (1032 * 1032 * 1032)<< "GB\n";
      // std::cout << "Used:"<< (sizeof(DATATYPE) * TOTAL_USED * 1.0) / (1032 * 1032 * 1032) << "GB\n";
    }
    // Todo instead of raw pointer in data use shared memory
    // This is a local change, add more tests to testfile testss/device_vector.cu

  //  std::shared_ptr<cuda_memory> d;
   // DATATYPE *d = nullptr;

   void fill(DATATYPE d);

   device_vector();

   device_vector(std::vector<DATATYPE>   &host);

   void resize(unsigned long  new_size);

   inline size_t capacity(){
     return allocated;
   }

   inline size_t size(){
     return current_size;
   }
   inline void clear(){
    TOTAL_USED -= current_size;
     free_size = allocated-current_size;
     resize(0);
   }

   void debug(std::string str);

    device_vector<DATATYPE>& clone(device_vector<DATATYPE> &in);
  
   ~device_vector();

   inline DATATYPE operator[](size_t id){
      assert(id < current_size);
      DATATYPE t;
      cudaMemcpy(&t,&((DATATYPE *)(this->data.data_ptr()))[id], sizeof(DATATYPE), cudaMemcpyDeviceToHost);
      return t;
   }

  
  device_vector<DATATYPE>& operator=(device_vector<DATATYPE>  &in);


  std::vector<DATATYPE> to_std_vector(){
      std::vector<DATATYPE> local;
      if (this->current_size == 0) return std::move(local);
      local.resize(this->current_size);
      cudaMemcpy(local.data(), this->data.data_ptr(), sizeof(DATATYPE) * this->current_size, cudaMemcpyDeviceToHost);
      return std::move(local);
  }

  void append(device_vector<DATATYPE> &in);

  // bool is_same(std::vector<DATATYPE> &expected);

  void resize_and_zero(int new_size);

  bool has_duplicates(){
    auto v1 = this->to_std_vector();
    std::sort(v1.begin(), v1.end());
     auto last = std::unique(v1.begin(), v1.end());
    v1.erase(last, v1.end());
    return !(v1.size() == this->size());
  }

  inline DATATYPE * ptr(){

    if(current_size ==0) return nullptr;
    assert(current_size > 0);
    return (DATATYPE *)this->data.data_ptr();
  }
  
  static void setLocalDevice(int id){
      device_vector<DATATYPE>::DEVICE  = id;
  }
 };

  
}
