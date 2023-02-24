#include "cub.h"


namespace cuslicer{

  device_vector<long> transform::d_temp_storage;
  device_vector<long> transform::d_temp_out;
  device_vector<long> transform::temporary;

long transform::reduce(cuslicer::device_vector<long> & data_d){
    assert(data_d.size() != 0);
    d_temp_out.resize(1);

    int num_elements = data_d.size();
    // Determine temporary device storage requirements
    size_t temp_storage_bytes;
    gpuErrchk(cub::DeviceReduce::Sum(NULL, temp_storage_bytes, data_d.ptr(), d_temp_out.ptr(), num_elements));
    d_temp_storage.resize(temp_storage_bytes/(sizeof(long)) + 1);
    gpuErrchk(cub::DeviceReduce::Sum(d_temp_storage.ptr(), temp_storage_bytes, data_d.ptr(), d_temp_out.ptr(), num_elements));
    return d_temp_out[0];
}


void transform::sort(cuslicer::device_vector<long> &in, cuslicer::device_vector<long> &out){
        assert(in.size() != 0  );
        int  num_items = in.size();          // e.g., 7
        long  *d_keys_in = in.ptr();         // e.g., [8, 6, 7, 5, 3, 0, 9]
        out.resize(num_items);
        long  *d_keys_out = out.ptr();        // e.g., [        ...        ]
        // Determine temporary device storage requirements
        size_t   temp_storage_bytes = 0;
        gpuErrchk(cub::DeviceRadixSort::SortKeys(NULL, temp_storage_bytes, d_keys_in, d_keys_out, num_items));

        d_temp_storage.resize(temp_storage_bytes/(sizeof(long)) + 1);
        // Run sorting operation
        gpuErrchk(cub::DeviceRadixSort::SortKeys(d_temp_storage.ptr(), temp_storage_bytes, d_keys_in, d_keys_out, num_items));
  }

void transform::unique(cuslicer::device_vector<long>& sorted_in, cuslicer::device_vector<long> & out){
    assert(sorted_in.size() != 0);
    // Declare, allocate, and initialize device-accessible pointers for input and output
    int  num_items = sorted_in.size();              // e.g., 8
    long  *d_in = sorted_in.ptr();                  // e.g., [0, 2, 2, 9, 5, 5, 5, 8]
    out.resize(num_items);
    long  *d_out = out.ptr();                 // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
    d_temp_out.resize(1);
    long  *d_num_selected_out = d_temp_out.ptr();    // e.g., [ ]

    // Determine temporary device storage requirements
    size_t   temp_storage_bytes = 0;
    gpuErrchk(cub::DeviceSelect::Unique(NULL, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items));
    d_temp_storage.resize(temp_storage_bytes/(sizeof(long)) + 1);
    // Run selection
    gpuErrchk(cub::DeviceSelect::Unique(d_temp_storage.ptr(), temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items));
    out.resize(d_temp_out[0]);
    // d_out                 <-- [0, 2, 9, 5, 8]
    // d_num_selected_out    <-- [5]
  }

void transform::remove_duplicates(cuslicer::device_vector<long> &in, cuslicer::device_vector<long>& out){
      sort(in, temporary);
      unique(temporary, out);
  }

void transform::exclusive_scan(cuslicer::device_vector<long> &in, cuslicer::device_vector<long>& out){
    assert(in.size() != 0);
    int  num_items = in.size();      // e.g., 7
    long  *d_in = in.ptr();          // e.g., [8, 6, 7, 5, 3, 0, 9]
    out.resize(num_items);
    long  *d_out = out.ptr();         // e.g., [ ,  ,  ,  ,  ,  ,  ]
    // Determine temporary device storage requirements
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(NULL, temp_storage_bytes, d_in, d_out, num_items);
    d_temp_storage.resize(temp_storage_bytes/(sizeof(long)) + 1);
    // Allocate temporary storage
    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage.ptr(), temp_storage_bytes, d_in, d_out, num_items);
  }

  void transform::inclusive_scan(cuslicer::device_vector<long> &in, cuslicer::device_vector<long>& out){
      assert(in.size() != 0);
      int  num_items = in.size();      // e.g., 7
      long  *d_in = in.ptr();          // e.g., [8, 6, 7, 5, 3, 0, 9]
      out.resize(num_items);
      long  *d_out = out.ptr();         // e.g., [ ,  ,  ,  ,  ,  ,  ]
      // Determine temporary device storage requirements
      size_t   temp_storage_bytes = 0;
      cub::DeviceScan::InclusiveSum(NULL, temp_storage_bytes, d_in, d_out, num_items);
      d_temp_storage.resize(temp_storage_bytes/(sizeof(long)) + 1);
      // Allocate temporary storage
      // Run exclusive prefix sum
      cub::DeviceScan::InclusiveSum(d_temp_storage.ptr(), temp_storage_bytes, d_in, d_out, num_items);
    }

    void transform::self_inclusive_scan(cuslicer::device_vector<long> &in){
        assert(in.size() != 0);
        int  num_items = in.size();      // e.g., 7
        long  *d_in = in.ptr();          // e.g., [8, 6, 7, 5, 3, 0, 9]
        long  *d_out = in.ptr();         // e.g., [ ,  ,  ,  ,  ,  ,  ]
        // Determine temporary device storage requirements
        size_t   temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(NULL, temp_storage_bytes, d_in, d_out, num_items);
        d_temp_storage.resize(temp_storage_bytes/(sizeof(long)) + 1);
        // Allocate temporary storage
        // Run exclusive prefix sum
        cub::DeviceScan::InclusiveSum(d_temp_storage.ptr(), temp_storage_bytes, d_in, d_out, num_items);
      }

      // Use Templates
      void transform::self_inclusive_scan_int(cuslicer::device_vector<int> &in){
          assert(in.size() != 0);
          int  num_items = in.size();      // e.g., 7
          int  *d_in = in.ptr();          // e.g., [8, 6, 7, 5, 3, 0, 9]
          int  *d_out = in.ptr();         // e.g., [ ,  ,  ,  ,  ,  ,  ]
          // Determine temporary device storage requirements
          size_t   temp_storage_bytes = 0;
          cub::DeviceScan::InclusiveSum(NULL, temp_storage_bytes, d_in, d_out, num_items);
          d_temp_storage.resize(temp_storage_bytes/(sizeof(int)) + 1);
          // Allocate temporary storage
          // Run exclusive prefix sum
          cub::DeviceScan::InclusiveSum(d_temp_storage.ptr(), temp_storage_bytes, d_in, d_out, num_items);
        }

}
