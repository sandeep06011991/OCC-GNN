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


class Stats{
  std::string name;
  std::string partition;
  std::shared_ptr<Dataset> dataset;
  NeighbourSampler *neighbour_sampler;
  Sample sample = Sample(3);
  vector<int> **layer_color;
  vector<int> workload_map;
public:
  Stats(const std::string& name, const std::string& partition, int fanout){
    populate_meta_dict();
    this->name = get_dataset_dir() + name;

    this->dataset = std::make_shared<Dataset>(this->name,false);
    std::cout << "Stats started\n";
    if(partition == "occ"){
      for(long j=0;j<dataset->num_nodes;j++){
        workload_map.push_back(dataset->partition_map[j]);
      }
    }

    std::cout << "Stats started\n";
    layer_color = (vector<int> **)malloc(4 * sizeof(vector<int> *));
    for(int i=0;i<4;i++){
      layer_color[i] = new vector<int>();
      for(long j=0;j<dataset->num_nodes;j++){
            layer_color[i]->push_back(0);
      }
    }

    std::cout << "Stats started\n";

    this->neighbour_sampler = new NeighbourSampler(this->dataset, fanout, false);

    std::cout << "Stats started\n";
  }

 pyredundant  get_stats(vector<long> sample_nodes){
    sample.clear();
    this->neighbour_sampler->sample(sample_nodes, sample);
    std::cout << "gat stats \n";
    redundant r = print_statistics(sample, layer_color, dataset->num_nodes, workload_map);
    std::cout << "got stats \n";

    pyredundant pr;
    pr.total_computation = r.total_computation;
    pr.redundant_compution = r.redundant_computation;
    pr.total_communication = r.total_communication;
    pr.redundant_communication = r.redundant_communication;
    std::cout << "total compuitation" << pr.total_computation <<"\n";
    return pr;
  }

  ~Stats(){
    for(int i=0;i<4;i++){
      delete layer_color[i];
    }
    free(layer_color);
  }
};

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
    bool self_edge;
public:
    // py::list v;

    CSlicer(const std::string &name,
      std::vector<std::vector<long>> gpu_map,
      int fanout,
       bool deterministic, bool testing, bool self_edge){
        spdlog::set_pattern("[%H:%M:%S %z] [%^%L%$] [thread %t] %v");
        spdlog::info("This an info message with custom format");
        spdlog::set_pattern("%+"); // back to default format
        spdlog::set_level(spdlog::level::info);
        // auto logger = spdlog::basic_logger_mt("test_logger" + std::rand()/1000, "logs/test.txt" + std::rand()/100);
        // spdlog::set_default_logger(logger);
        // spdlog::flush_on(spdlog::level::info);
        spdlog::info("Log after checking");
        // spdlog::get("test_logger")->info("LoggingTest::ctor");

        this->name = get_dataset_dir() + name;
        // std::cout << this->name << "\n";
        this->deterministic = deterministic;

        this->dataset = std::make_shared<Dataset>(this->name, testing);
        std::cout <<" This is a testing dataset " << testing << "\n";
        spdlog::info("Log after checking the dataset");
        num_nodes = dataset->num_nodes;
	      std::cout << "Read graph with number of nodes: " << num_nodes <<"\n";
        this->self_edge = self_edge;
        for(long j=0;j<dataset->num_nodes;j++){
            assert(dataset->partition_map[j]<4);
            workload_map.push_back(dataset->partition_map[j]);
            for(int k=0;k<4;k++){
              storage_map[k].push_back(-1);
            }
        }
        assert(False);
        // Dont exapnd here storage map is undordered.
        
        for(int i=0;i<4;i++){
          int j = 0;
          int order =0;
          dummy_storage_map[i].clear();
          gpu_capacity[i] = gpu_map[i].size();
            for(long nd: gpu_map[i]){
            storage_map[i][nd] = order;
            order ++;
            dummy_storage_map[i].push_back(nd);
          }
        }

        this->slicer = new Slice((workload_map), storage_map, self_edge);
        this->neighbour_sampler = new NeighbourSampler(this->dataset, fanout, deterministic);
    }

    bool test_correctness(vector<long> sample_nodes){
      sample.clear();
      p_sample.clear();
      this->neighbour_sampler->sample(sample_nodes, sample);
      int sample_val =  sample_flow_up_sample(sample, num_nodes);
      this->slicer->slice_sample(sample, p_sample);
      this->slicer->measure_pull_benefits(sample);
      // spdlog::info("covert to torch");
      PySample *sample = new PySample(p_sample);
      std::vector<int> ret(4);
      for(int i=0;i<4;i++){
        ret[i] = 0;
      }
      int p_val =  sample_flow_up_ps(p_sample, dummy_storage_map, ret);
      return (sample_val  == p_val);
    }

    PySample * getSample(vector<long> sample_nodes){
      sample.clear();
      p_sample.clear();
      // spdlog::info("sample begin");
      this->neighbour_sampler->sample(sample_nodes, sample);
      int sample_val;
      if(this->deterministic and !this->self_edge){
          sample_val =  sample_flow_up_sample(sample, num_nodes);
      }
      // spdlog::info("slice begin");
      this->slicer->slice_sample(sample, p_sample);
      // spdlog::info("covert to torch");
      PySample *sample = new PySample(p_sample);
      std::vector<int> ret(4);
      for(int i=0;i<4;i++){
        ret[i] = 0;
      }

      // if(this->deterministic){
      if(this->deterministic and !this->self_edge){
        std::cout << "debug";
          // for(int i=0;i<4;i++){
          //   vector<long> &m = p_sample.refresh_map[i];
          //   for(long n:m){
          //     dummy_storage_map[i].push_back(n);
          //   }
          // }
          int p_val =  sample_flow_up_ps(p_sample, dummy_storage_map, ret);
          // for(int i=0;i<4;i++){
          //   int s = p_sample.refresh_map[i].size();
          //   int cs = dummy_storage_map[i].size();
          //   assert(cs > = s);
          //   dummy_storage_map[i].resize(cs-s);
          // }
          std::cout << "My anser is " << p_val << "sample_val "<< sample_val << "\n";
          assert(sample_val  == p_val);
      }
      sample->debug_vals = ret;
      // std::cout << "I have a sample\n";
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
               std::vector<std::vector<long>>, int, bool, bool,bool>())
         .def("getSample", &CSlicer::getSample, py::return_value_policy::take_ownership)\
         .def("sampleAndVerify",&CSlicer::test_correctness);
         py::class_<PySample>(m,"sample")
             .def_readwrite("layers",&PySample::layers)
             .def_readwrite("in_nodes", &PySample::in_nodes)
             .def_readwrite("out_nodes", &PySample::out_nodes)
             .def_readwrite("missing_node_ids", &PySample::missing_node_ids)
             .def_readwrite("cached_node_ids", &PySample::cached_node_ids)
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

        py::class_<Stats>(m,"stats")
          .def(py::init<const std::string &,
                const std::string &, int>())
          .def("get_stats", &Stats::get_stats);

        py::class_<pyredundant>(m, "redundant")
          .def_readwrite("total_computation", &pyredundant::total_computation)
          .def_readwrite("redundant_compution", &pyredundant::redundant_compution)
          .def_readwrite("total_communication",&pyredundant::total_communication)
         .def_readwrite("redundant_communication",&pyredundant::redundant_communication);

}
