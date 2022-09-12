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
namespace py = pybind11;


class CSlicer{
    std::string name;
    int samples_generated = 0;
    long num_nodes;

    std::vector<int> storage_map[4];
    std::vector<int> workload_map;
    int gpu_capacity[4];
    NeighbourSampler *neighbour_sampler;
    Slice *slicer;
    std::shared_ptr<Dataset> dataset;
    PartitionedSample p_sample;
    Sample sample = Sample(3);
public:
    // py::list v;

    CSlicer(const std::string &name,
      std::vector<std::vector<long>> gpu_map,
      int fanout,
       bool deterministic){
        spdlog::info("Running {}",name);
        this->name = get_dataset_dir() + name;
        // std::cout << this->name << "\n";

        this->dataset = std::make_shared<Dataset>(this->name);
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
          }
        }

        this->slicer = new Slice((workload_map), storage_map);
        this->neighbour_sampler = new NeighbourSampler(this->dataset, fanout);
    }

    PySample * getSample(vector<long> sample_nodes){
      sample.clear();
      p_sample.clear();
      this->neighbour_sampler->sample(sample_nodes, sample);
      this->slicer->slice_sample(sample, p_sample);
      PySample *sample = new PySample(p_sample);
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
             .def_readwrite("missing_node_ids", &PySample::missing_node_ids);
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
