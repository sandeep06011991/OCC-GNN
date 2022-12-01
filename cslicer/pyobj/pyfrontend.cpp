#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <string>
#include "pybipartite.h"
#include "graph/dataset.h"
#include "transform/slice.h"
#include "util/environment.h"
#include "samplers/samplers.h"
#include "graph/sample.h"
#include <cstdlib>
#include <iostream>
#include <ctime>
namespace py = pybind11;

int sample_flow_up_sample(Sample &s, int number_of_nodes);

int sample_flow_up_ps(PartitionedSample &s,
    std::vector<int> test_storage_map[4], std::vector<int>& ret);

struct pyredundant{
  int total_computation = 0;
  int redundant_compution = 0;
  int total_communication = 0;
  int redundant_communication = 0;
};

// FixME.
// Current stats methodology is wrong fix it later.
// class Stats{
//   std::string name;
//   std::string partition;
//   std::shared_ptr<Dataset> dataset;
//   NeighbourSampler *neighbour_sampler;
//   Sample sample = Sample(3);
//   vector<int> **layer_color;
//   vector<int> workload_map;
// public:
//   Stats(const std::string& name, const std::string& partition, int fanout){
//     populate_meta_dict();
//     this->name = get_dataset_dir() + name;
//
//     this->dataset = std::make_shared<Dataset>(this->name,false);
//     if(partition == "occ"){
//       for(long j=0;j<dataset->num_nodes;j++){
//         workload_map.push_back(dataset->partition_map[j]);
//       }
//     }
//     layer_color = (vector<int> **)malloc(4 * sizeof(vector<int> *));
//     for(int i=0;i<4;i++){
//       layer_color[i] = new vector<int>();
//       for(long j=0;j<dataset->num_nodes;j++){
//             layer_color[i]->push_back(0);
//       }
//     }
//     this->neighbour_sampler = new NeighbourSampler(this->dataset, fanout, false);
//   }
//
//  pyredundant  get_stats(vector<long> sample_nodes){
//     sample.clear();
//     this->neighbour_sampler->sample(sample_nodes, sample);
//     redundant r = print_statistics(sample, layer_color, dataset->num_nodes, workload_map);
//     pyredundant pr;
//     pr.total_computation = r.total_computation;
//     pr.redundant_compution = r.redundant_computation;
//     pr.total_communication = r.total_communication;
//     pr.redundant_communication = r.redundant_communication;
//     std::cout << "total compuitation" << pr.total_computation <<"\n";
//     return pr;
//   }
//
//   ~Stats(){
//     for(int i=0;i<4;i++){
//       delete layer_color[i];
//     }
//     free(layer_color);
//   }
// };

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
    PartitionedSample *p_sample;
    Sample *sample;
    bool deterministic;
    bool self_edge;
public:
    // py::list v;

    CSlicer(const std::string &name,
      std::vector<std::vector<long>> gpu_map,
      int fanout,
       bool deterministic, bool testing,
          bool self_edge, int rounds, bool pull_optimization, int num_layers){

        this->name = get_dataset_dir() + name;
        // std::cout << this->name << "\n";
        this->deterministic = deterministic;

        this->dataset = std::make_shared<Dataset>(this->name, testing);

        num_nodes = dataset->num_nodes;

        this->self_edge = self_edge;
        for(long j=0;j<dataset->num_nodes;j++){
            assert(dataset->partition_map[j]<4);
            workload_map.push_back(dataset->partition_map[j]);
        }

        for(int i=0;i<4;i++){
          int order =0;
          for(long nd: gpu_map[i]){
             storage_map[i].push_back(nd);
             order ++;
          }
          gpu_capacity[i] = gpu_map[i].size();
        }
	this->sample = new Sample(num_layers);
	this->p_sample = new PartitionedSample(num_layers);
        this->slicer = new Slice((workload_map), storage_map, self_edge, rounds, pull_optimization);
        this->neighbour_sampler = new NeighbourSampler(this->dataset, fanout, deterministic, self_edge);
    }

    bool test_correctness(vector<long> sample_nodes){
      sample->clear();
      p_sample->clear();
      this->neighbour_sampler->sample(sample_nodes, *sample);
      this->slicer->slice_sample(*sample, *p_sample);
      // spdlog::info("covert to torch");
      PySample *sample = new PySample(*p_sample);
    }

    PySample * getSample(vector<long> sample_nodes){
      sample->clear();
      p_sample->clear();
      // spdlog::info("sample begin");
      this->neighbour_sampler->sample(sample_nodes, *sample);
      // spdlog::info("slice begin");
      this->slicer->slice_sample(*sample, *p_sample);
      // spdlog::info("covert to torch");
      PySample *sample = new PySample(*p_sample);
      return sample;
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
               std::vector<std::vector<long>>, int, bool, bool, bool, int, bool,int >())
         .def("getSample", &CSlicer::getSample, py::return_value_policy::take_ownership)\
         .def("sampleAndVerify",&CSlicer::test_correctness);
         py::class_<PySample>(m,"sample")
             .def_readwrite("layers",&PySample::layers)
             .def_readwrite("cache_hit_from", &PySample::cache_hit_from)
             .def_readwrite("cache_hit_to", &PySample::cache_hit_to)
             .def_readwrite("cache_miss_from", &PySample::cache_miss_from)
             .def_readwrite("cache_miss_to", &PySample::cache_miss_to)
             .def_readwrite("out_nodes", &PySample::out_nodes)
             .def_readwrite("debug_vals", &PySample::debug_vals);
         py::class_<PyBipartite>(m,"bipartite")
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
