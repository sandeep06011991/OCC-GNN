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
namespace cuslicer{

  template <typename DATATYPE>
  class device_vector{

    size_t allocated = 0;

    size_t free_size= 0;


    class cuda_memory{



      public:

        DATATYPE *data = nullptr;

        cuda_memory(DATATYPE *d){
            this->data = d;
          }

          ~cuda_memory(){
            gpuErrchk(cudaFree(data));
          }

          static std::shared_ptr<cuda_memory> alloc(int size_t){
              DATATYPE *d;
              gpuErrchk(cudaMalloc((void**)&d, (sizeof(DATATYPE) * size_t)));
              return std::make_shared<cuda_memory>(d);
          }

          inline DATATYPE * ptr(){return data;}
    };

  public:
    size_t current_size= 0;

    // Todo instead of raw pointer in data use shared memory
    // This is a local change, add more tests to testfile testss/device_vector.cu

   std::shared_ptr<cuda_memory> d;
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
     current_size = 0;
     free_size = allocated-current_size;
   }

   void debug(std::string str);

   inline DATATYPE * ptr(){
     if(current_size ==0) return nullptr;
     return d->ptr();
   }

   ~device_vector();

   inline DATATYPE operator[](size_t id){
      assert(id < current_size);
      DATATYPE t;
      cudaMemcpy(&t,&(d->ptr()[id]), sizeof(DATATYPE), cudaMemcpyDeviceToHost);
      return t;
   }

  inline void set_value(size_t id, DATATYPE val){
     assert(id < current_size);
     DATATYPE t = val;
     cudaMemcpy(&(d->ptr()[id]), &t, sizeof(DATATYPE), cudaMemcpyHostToDevice);
  }

  inline void destroy(){
    current_size = 0;
    allocated = 0;
    free_size = 0;
    d = nullptr;
  }

  device_vector<DATATYPE>& operator=(device_vector<DATATYPE>  &in);

  // device_vector<DATATYPE> operator=(device_vector<DATATYPE> in);


  void append(device_vector<DATATYPE> &in);

  bool is_same(std::vector<DATATYPE> &expected);

  void resize_and_zero(int new_size){
    this->resize(new_size);
    this->fill(0);
  }

 };

}
