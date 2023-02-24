#pragma once
#include <cub/cub.cuh>
#include "cuda_utils.h"
#include "device_vector.h"

namespace cuslicer{
  // Add using stream
  class transform{

  private:
    static cuslicer::device_vector<long> d_temp_storage;
    static cuslicer::device_vector<long> d_temp_out;
    static cuslicer::device_vector<long> temporary;

  transform(){}
  public:

    static long reduce(cuslicer::device_vector<long>& data_d);

    static void sort(cuslicer::device_vector<long> &in, cuslicer::device_vector<long> &out);

    static void unique(cuslicer::device_vector<long>& sorted_in, cuslicer::device_vector<long> & out);

    static void remove_duplicates(cuslicer::device_vector<long> &in, cuslicer::device_vector<long>& out);

    static void exclusive_scan(cuslicer::device_vector<long> &in, cuslicer::device_vector<long>& out);

    static void inclusive_scan(cuslicer::device_vector<long> &in, cuslicer::device_vector<long>& out);

    static void self_inclusive_scan(cuslicer::device_vector<long> &in);

  static void  self_inclusive_scan_int(cuslicer::device_vector<int> &in);

    static void cleanup(){
        d_temp_storage.destroy();
        d_temp_out.destroy();
        temporary.destroy();
    }

  };


}
