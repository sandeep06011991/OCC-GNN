#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <string>
#include "pybipartite.h"
#include "graph/dataset.h"
#include "transform/slice.h"
#include "util/environment.h"
#include "spdlog/spdlog.h"
#include "samplers/samplers.h"
#include "graph/sample.h"
#include "spdlog/sinks/basic_file_sink.h"
namespace py = pybind11;

int sample_flow_up_sample(Sample &s, int number_of_nodes);

int sample_flow_up_ps(PartitionedSample &s,
    std::vector<int> test_storage_map[4], std::vector<int>& ret);

class CSlicer{
    std::string name;
    int samples_generated = 0;
    long num_nodes;

    std::vector<int> storage_map[4];
    std::vector<int> workload_map;
    std::vector<int> dummy_storage_map[4];
    int gpu_capacity[4];
    NeighbourSampler *neighbour_sampler;
    Slice *slicer;
    std::shared_ptr<Dataset> dataset;
    PartitionedSample p_sample;
    Sample sample = Sample(3);
    bool deterministic;
    // static const auto my_logger = spdlog::basic_logger_mt("file_logger", "logs/basic-log.txt", true);

public:
    // py::list v;

    CSlicer(const std::string &name,
      std::vector<std::vector<long>> gpu_map,
      int fanout,
       bool deterministic){
        spdlog::set_pattern("[%H:%M:%S %z] [%^%L%$] [thread %t] %v");
        spdlog::info("This an info message with custom format");
        spdlog::set_pattern("%+"); // back to default format
        spdlog::set_level(spdlog::level::info);
        auto logger = spdlog::basic_logger_mt("test_logger", "logs/test.txt");
        spdlog::set_default_logger(logger);
        spdlog::flush_on(spdlog::level::info);
        spdlog::info("Log after checking");
        spdlog::get("test_logger")->info("LoggingTest::ctor");
        std::cout << "Check number of nodes " << num_nodes <<"\n";
        this->name = get_dataset_dir() + name;
        // std::cout << this->name << "\n";
        this->deterministic = deterministic;

        this->dataset = std::make_shared<Dataset>(this->name);
        spdlog::info("Log after checking the dataset");
        num_nodes = dataset->num_nodes;


        for(long j=0;j<dataset->num_nodes;j++){
            assert(dataset->partition_map[j]<4);
            workload_map.push_back(dataset->partition_map[j]);
            for(int k=0;k<4;k++){
              storage_map[k].push_back(-1);
            }
        }

        std::cout << "Printing GPU local to global map \n";
        for(int i=0;i<4;i++){
          int j = 0;
          int order =0;
          gpu_capacity[i] = gpu_map[i].size();
          for(long nd: gpu_map[i]){
            storage_map[i][nd] = order;
            order ++;
            dummy_storage_map[i].push_back(nd);
          }
        }

        this->slicer = new Slice((workload_map), storage_map);
        this->neighbour_sampler = new NeighbourSampler(this->dataset, fanout);
    }

    PySample * getSample(vector<long> sample_nodes){
      sample.clear();
      p_sample.clear();
      spdlog::info("sample begin");
      this->neighbour_sampler->sample(sample_nodes, sample);
      int sample_val;
      if(this->deterministic){
          sample_val =  sample_flow_up_sample(sample, num_nodes);
      }

      spdlog::info("slice begin");
      this->slicer->slice_sample(sample, p_sample);
      spdlog::info("covert to torch");
      PySample *sample = new PySample(p_sample);
      std::vector<int> ret(4);
      for(int i=0;i<4;i++){
        ret[i] = 0;
      }
      if(this->deterministic){
          int p_val =  sample_flow_up_ps(p_sample, dummy_storage_map, ret);
          std::cout << "My anser is " << p_val << "sample_val "<< "\n";
          assert(sample_val  == p_val);
      }
      sample->debug_vals = ret;

      // p_sample.debug();
      return sample;
      // Sample *s = Sample::get_dummy_sample();
      // return new PySample(s);
      //std::cout << "Try to get a sample \n";
      // return this->pool->pop_object();
    }

    ~CSlicer(){
      std::cout <<"cslicer clean up start\n";
      // Delete dataset
      // Delete cslicer
    }
};

PYBIND11_MODULE(cslicer, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    py::class_<CSlicer>(m,"cslicer")
         .def(py::init<const std::string &,
               std::vector<std::vector<long>>, int, bool>())
         .def("getSample", &CSlicer::getSample, py::return_value_policy::take_ownership);
         py::class_<PySample>(m,"sample")
             .def_readwrite("layers",&PySample::layers)
             .def_readwrite("in_nodes", &PySample::in_nodes)
             .def_readwrite("out_nodes", &PySample::out_nodes)
             .def_readwrite("missing_node_ids", &PySample::missing_node_ids)
              .def_readwrite("debug_vals", &PySample::debug_vals);
         py::class_<PyBipartite>(m,"bipartite")
             .def_readwrite("num_in_nodes", &PyBipartite::num_in_nodes)
             .def_readwrite("num_out_nodes", &PyBipartite::num_out_nodes)
             .def_readwrite("in_nodes",&PyBipartite::in_nodes)
     	      .def_readwrite("indptr",&PyBipartite::indptr)
     	      .def_readwrite("expand_indptr", &PyBipartite::expand_indptr)
     	       .def_readwrite("out_nodes",&PyBipartite::out_nodes)
             .def_readwrite("owned_out_nodes",&PyBipartite::owned_out_nodes)
     	      .def_readwrite("indices",&PyBipartite::indices)
     	      .def_readwrite("from_ids",&PyBipartite::from_ids)
     	      .def_readwrite("to_ids",&PyBipartite::to_ids)
     	      .def_readwrite("self_ids_in",&PyBipartite::self_ids_in)
     	      .def_readwrite("self_ids_out",&PyBipartite::self_ids_out)
     	       .def_readwrite("indegree", &PyBipartite::indegree)
              .def_readwrite("gpu_id",&PyBipartite::gpu_id);
}
