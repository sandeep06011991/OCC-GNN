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
namespace cuslicer{

  template <typename DATATYPE>
  class device_vector{

    size_t allocated = 0;

    size_t free_size= 0;

  public:
    size_t current_size= 0;
    DATATYPE *d = nullptr;


    device_vector();

   device_vector(std::vector<DATATYPE> &host);

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
     return d;
   }

   ~device_vector();

   inline DATATYPE operator[](size_t id){
      assert(id < current_size);
      DATATYPE t;
      cudaMemcpy(&t,&d[id], sizeof(DATATYPE), cudaMemcpyDeviceToHost);
      return t;
   }

  inline void set_value(size_t id, DATATYPE val){
     assert(id < current_size);
     DATATYPE t = val;
     cudaMemcpy(&d[id], &t, sizeof(DATATYPE), cudaMemcpyHostToDevice);
  }

  inline void destroy(){
    current_size = 0;
    allocated = 0;
    free_size = 0;
    if(d != nullptr)cudaFree(d);
    d = nullptr;
  }

  void operator=(device_vector<DATATYPE> &in);

  void append(device_vector<DATATYPE> &in);

  bool is_same(std::vector<DATATYPE> &expected);
 };

}
