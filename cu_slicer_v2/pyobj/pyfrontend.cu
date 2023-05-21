#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <string>
#include "pybipartite.h"
#include "../graph/dataset.cuh"
#include "../transform/slice.h"
#include "../util/environment.h"
#include "../samplers/samplers.h"
#include "../graph/sample.h"
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <chrono>
#include "../util/cuda_utils.h"
#include "../util/cub.h"
#include "../util/types.h"
#include "../util/device_vector.h"

#include <memory>
using namespace std::chrono;
namespace py = pybind11;

__global__
void testKernel(NDTYPE *t){
  t[0] = 0;
}
int sample_flow_up_sample(Sample &s, int number_of_nodes);

int sample_flow_up_ps(PartitionedSample &s,
    std::vector<int> test_storage_map[4], std::vector<int>& ret);

struct pyredundant{
  int total_computation = 0;
  int redundant_compution = 0;
  int total_communication = 0;
  int redundant_communication = 0;
};

class CUSlicer{
    std::string name;
    int samples_generated = 0;
    long num_nodes;

    std::vector<NDTYPE> storage_map[MAX_DEVICES];
    std::vector<PARTITIONIDX> workload_map;
    int gpu_capacity[MAX_DEVICES];
    NeighbourSampler *neighbour_sampler;
    Slice *slicer;
    std::shared_ptr<Dataset> dataset;
    PartitionedSample *p_sample;
    Sample *sample;
    bool deterministic;
    bool self_edge;
    int num_gpus = -1;
    int current_gpu = -1;

public:
    // py::list v;
    CUSlicer(const std::string &name,
      std::vector<std::vector<NDTYPE>> gpu_map,
      vector<int> fanout,
       bool deterministic, bool testing,
          bool self_edge, int rounds, bool pull_optimization,
            int num_layers, int num_gpus, int current_gpu, bool random, bool UVA){
        this->num_gpus = num_gpus;
        this->current_gpu = current_gpu;
        cudaSetDevice(current_gpu);
        this->name = get_dataset_dir() + name;
        std::cout << "Got dataset" << this->name << "\n";
        std::cout << "Use UVA" << UVA <<"\n";
        this->deterministic = deterministic;
        this->dataset = std::make_shared<Dataset>(this->name,  num_gpus, random, UVA);

        num_nodes = dataset->num_nodes;

        this->self_edge = self_edge;
        std::cout << "Start popilation\n";

        workload_map = dataset->partition_map_d.to_std_vector();

        std::vector<NDTYPE> _t;
        std::cout << "begin data populatiopn\n";
        for(int i=0;i<num_gpus;i++){
          int order =0;
          _t.clear();
          for(auto nd: gpu_map[i]){
             _t.push_back(nd);
             order ++;
          }
          storage_map[i] = _t;
          gpu_capacity[i] = gpu_map[i].size();
        }
        std::cout << "data opulated\n";
      	this->sample = new Sample(num_layers);
        std::cout << "Sampling Layers \n";
      	this->p_sample = new PartitionedSample(num_layers, num_gpus);
        bool load = false;
        std::cout << "Sampling Layers Cross \n";

        this->neighbour_sampler = new NeighbourSampler(this->dataset, fanout,  self_edge);
        this->slicer = new PullSlicer((workload_map), storage_map,  pull_optimization, num_gpus, \
              this->neighbour_sampler->dev_curand_states);
        std::cout << "Checl again \n";
    }

    // bool test_correctness(vector<long> sample_nodes){
    //   sample->clear();
    //   p_sample->clear();
    //   this->neighbour_sampler->sample(sample_nodes, *sample);
    //   this->slicer->slice_sample(*sample, *p_sample);
    //   // spdlog::info("covert to torch");
    //   PySample *sample = new PySample(*p_sample);
    // }

    unique_ptr<PySample> getSample(vector<NDTYPE> sample_nodes, 
      bool balance){
      std::cout << "try to get a sample \n";
      sample->clear();
      p_sample->clear();
      cudaSetDevice(current_gpu);
      auto start1 = high_resolution_clock::now();

      cuslicer::device_vector<NDTYPE> sample_nodes_d(sample_nodes);
      this->neighbour_sampler->sample(sample_nodes_d, *sample);
      cudaDeviceSynchronize();
      auto start2 = high_resolution_clock::now();


      // spdlog::info("slice begin");
      std::cout << "attempting slicing \n";
      this->slicer->slice_sample(*sample, *p_sample, balance);
  	  cudaDeviceSynchronize();
      auto start3 = high_resolution_clock::now();
      auto duration1 = duration_cast<milliseconds>(start2 - start1);
      auto duration2 = duration_cast<milliseconds>(start3 -start2);

     std::cout << "sample " << (double)duration1.count()/1000 << "slice"<< (double)duration2.count()/1000 <<"\n";
      device_vector<NDTYPE>::printMemoryStats();
      // device_vector<long>::printMemoryStats();
      // spdlog::info("covert to torch");
      auto sample = std::make_unique<PySample>(*p_sample, current_gpu, num_gpus);
      return sample;
    }

    ~CUSlicer(){
      std::cout <<"cslicer clean up start\n";
      // Delete dataset
      // Delete cslicer
    }

    torch::Tensor getDummyTensor(){
        cudaSetDevice(0);
        std::vector<NDTYPE> data;
        for(int i = 0;i < 1000; i++){
          data.push_back(i);
        }
        auto v = device_vector<NDTYPE>(data);
        testKernel<<<1,1>>>(v.ptr());
        gpuErrchk(cudaDeviceSynchronize());
        auto sum = cuslicer::transform<NDTYPE>::reduce(v);

        std::cout << sum <<"sum \n";
        auto opts = torch::TensorOptions().dtype(torch::kInt32)\
        .device(torch::kCUDA, 0);
        return torch::from_blob(v.ptr(), {(long)v.size()}, opts).clone();
    }

};

PYBIND11_MODULE(cuslicer, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    py::class_<CUSlicer>(m,"cuslicer")
         .def(py::init<const std::string &,
               std::vector<std::vector<NDTYPE>>, vector<int>,\
                bool, bool, bool, int, bool,int,\
                  int, int, bool,bool>())
        .def("getTensor", &CUSlicer::getDummyTensor)
         .def("getSample", &CUSlicer::getSample, py::return_value_policy::take_ownership);
         // .def("sampleAndVerify",&CSlicer::test_correctness);
         py::class_<PySample>(m,"CUsample")
             .def_readwrite("layers",&PySample::layers)
             .def_readwrite("cache_hit_from", &PySample::cache_hit_from)
             .def_readwrite("cache_hit_to", &PySample::cache_hit_to)
             .def_readwrite("cache_miss_from", &PySample::cache_miss_from)
             .def_readwrite("cache_miss_to", &PySample::cache_miss_to)
             .def_readwrite("out_nodes", &PySample::out_nodes)
             .def_readwrite("debug_vals", &PySample::debug_vals);
         py::class_<PyBipartite>(m,"CUbipartite")
             .def_readwrite("num_in_nodes_local", &PyBipartite::num_in_nodes_local)
             .def_readwrite("num_in_nodes_pulled", &PyBipartite::num_in_nodes_pulled)
             .def_readwrite("num_out_local",&PyBipartite::num_out_local)
     	       .def_readwrite("num_out_remote",&PyBipartite::num_out_remote)
     	       .def_readwrite("out_degree_local", &PyBipartite::out_degree_local)
     	       .def_readwrite("indptr_L",&PyBipartite::indptr_L)
             .def_readwrite("indices_L",&PyBipartite::indices_L)
     	       .def_readwrite("indptr_R",&PyBipartite::indptr_R)
     	      .def_readwrite("indices_R",&PyBipartite::indices_R)
     	      .def_readwrite("from_ids",&PyBipartite::from_ids)
     	      .def_readwrite("push_to_ids",&PyBipartite::push_to_ids)
     	      .def_readwrite("to_offsets",&PyBipartite::to_offsets)
     	      .def_readwrite("pull_from_offsets", &PyBipartite::pull_from_offsets)
            .def_readwrite("self_ids_offset", &PyBipartite::self_ids_offset)
            .def_readwrite("gpu_id",&PyBipartite::gpu_id);

//          These metrics are wrong
        // py::class_<Stats>(m,"stats")
        //   .def(py::init<const std::string &,
        //         const std::string &, int>())
        //   .def("get_stats", &Stats::get_stats);
        //
        // py::class_<pyredundant>(m, "redundant")
        //   .def_readwrite("total_computation", &pyredundant::total_computation)
        //   .def_readwrite("redundant_compution", &pyredundant::redundant_compution)
        //   .def_readwrite("total_communication",&pyredundant::total_communication)
        //  .def_readwrite("redundant_communication",&pyredundant::redundant_communication);

}
