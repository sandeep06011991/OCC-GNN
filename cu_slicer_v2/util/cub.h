#pragma once
#include <cub/cub.cuh>
#include "cuda_utils.h"
#include "device_vector.h"
#include "../util/types.h"

namespace cuslicer{
  // Add using stream
  template<typename T>
  class transform{

  private:
    static cuslicer::device_vector<T> d_temp_storage;
    static cuslicer::device_vector<T> d_temp_out;
    static cuslicer::device_vector<T> temporary;

  transform(){}
  public:

    static T reduce(cuslicer::device_vector<T>& data_d);

    static void sort(cuslicer::device_vector<T> &in,\
           cuslicer::device_vector<T> &out);

    static void unique(cuslicer::device_vector<T>& sorted_in,\
         cuslicer::device_vector<T> & out);

    static void remove_duplicates(cuslicer::device_vector<T> &in,\
       cuslicer::device_vector<T>& out);

    static void exclusive_scan(cuslicer::device_vector<T> &in,\
       cuslicer::device_vector<T>& out);

    static void inclusive_scan(cuslicer::device_vector<T> &in,\
           cuslicer::device_vector<T>& out);

    static void self_inclusive_scan(cuslicer::device_vector<T> &in);

    // static void  self_inclusive_scan_int(cuslicer::device_vector<T> &in);

    static void cleanup(){
        d_temp_storage.destroy();
        d_temp_out.destroy();
        temporary.destroy();
    }

  };

  template class transform<NDTYPE>;
  // temptransform<int>;
}
