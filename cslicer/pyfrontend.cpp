#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <string>
#include "object.h"
#include "pybipartite.h"
#include "WorkerPool.h"
#include "dataset.h"

namespace py = pybind11;
// int add(int i, int j) {
//     return i + j;
// }
// struct Pet {
//     Pet(const std::string &name):name(name){}
//     void setName(const std::string &name_){name = name_;};
//     const std::string &getName() const{return name;}
//     std::string name;
// };
const string dir = "/data/sandeep/";

class CSlicer{
    std::string name;
    int queue_size;
    int no_worker_threads;
    int number_of_epochs;
    int minibatch_size;
    int samples_generated = 0;
    long num_nodes;
    WorkerPool * pool;
    std::vector<int> *storage_map[4];
    std::vector<int> *workload_map;
    int gpu_capacity[4];
    int capacity[4];
public:
    // py::list v;
    std::thread *th;

    CSlicer(const std::string &name, int queue_size, int no_worker_threads \
        , int number_of_epochs,  int minibatch_size, std::vector<std::vector<long>> gpu_map){
        this->name = dir + name;
        std::cout << this->name << "\n";
        this->queue_size = queue_size;
        this->no_worker_threads = no_worker_threads;
        this->number_of_epochs = number_of_epochs;
        this->minibatch_size = minibatch_size;

        Dataset * dataset = new Dataset(this->name);
        num_nodes = dataset->num_nodes;

        for(int i=0;i<4;i++){
          storage_map[i] = new std::vector<int>();
        }
        workload_map = new vector<int>();
        vector<int> temp;
        for(long j=0;j<dataset->num_nodes;j++){
            assert(dataset->partition_map[j]<4);
            workload_map->push_back(dataset->partition_map[j]);
            temp.push_back(-1);
          // workload_map.push_back(j%4);
          for(int k=0;k<4;k++){
              storage_map[k]->push_back(-1);
            }
        }
        for(int i=0;i<4;i++){
          int order =0;
          gpu_capacity[i] = gpu_map[i].size();
          for(long nd: gpu_map[i]){
            (*storage_map[i])[nd] = order;
            order ++;
            temp[nd] = 1;
          }
        }
        for(int i=0;i<dataset->num_nodes;i++){
          assert(temp[i] == 1);
          if(temp[i]!= 1){
            std::cout << "mising node " << i <<"\n";
          }
        }
        this->pool = new WorkerPool(num_nodes, number_of_epochs,
           minibatch_size, no_worker_threads, dataset, workload_map,
            storage_map, gpu_capacity);
        th = new std::thread(&WorkerPool::run, this->pool);
        std::cout << "pool is running \n";
    }

    PySample * getSample(){
      // Sample *s = Sample::get_dummy_sample();
      // return new PySample(s);
      std::cout << "Try to get a sample \n";
      return this->pool->pop_object();
    }

    long expected_number_of_samples(){
      long samples_per_epoch = (num_nodes - 1)/minibatch_size +1;
      return samples_per_epoch * number_of_epochs;
    }

    ~CSlicer(){
      std::cout <<"clean up \n";
      th->join();
    }
};

py::list testlist(std::vector<std::vector<int>> l) {
    // l.attr("pop")();
    // std::cout << "List has length " << l.size() << std::endl;
    // for (py::handle obj : l) {  // iterators!
    //     std::cout << "  - " << obj.attr("__str__")().cast<std::string>() << std::endl;
    // }
    std::vector<int> test_vec;
    test_vec.push_back(1);
    test_vec.push_back(2);
    test_vec.push_back(3);
    test_vec.push_back(4);
    // py::list test_list3 = py::cast(test_vec);
    l.push_back(test_vec);  // automatic casting (through templating)!
    return py::cast(l);
}

PySample * testpysample(){
    Sample *sh = new Sample();
    return new PySample(sh);
}

PYBIND11_MODULE(cslicer, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("test_list", &testlist, "List testing ");
    m.def("test_pyfront", &testpysample, "List testing ",py::return_value_policy::take_ownership);
    // m.def("add", &add, "A function that adds two numbers");
    py::class_<PySample>(m,"sample")
        .def_readwrite("layers",&PySample::layers)
        .def_readwrite("in_nodes", &PySample::in_nodes)
        .def_readwrite("out_nodes", &PySample::out_nodes);
    py::class_<PyBipartite>(m,"bipartite")
        .def_readwrite("num_in_nodes", &PyBipartite::num_in_nodes)
        .def_readwrite("num_out_nodes", &PyBipartite::num_out_nodes)
        .def_readwrite("in_nodes",&PyBipartite::in_nodes)
        .def_readwrite("indptr",&PyBipartite::indptr)
        .def_readwrite("out_nodes",&PyBipartite::out_nodes)
        .def_readwrite("owned_out_nodes",&PyBipartite::owned_out_nodes)
        .def_readwrite("indices",&PyBipartite::indices)
        .def_readwrite("from_ids",&PyBipartite::from_ids)
        .def_readwrite("to_ids",&PyBipartite::to_ids)
        .def_readwrite("self_ids_in",&PyBipartite::self_ids_in)
        .def_readwrite("self_ids_out",&PyBipartite::self_ids_out)
        .def_readwrite("gpu_id",&PyBipartite::gpu_id);

    py::class_<CSlicer>(m,"cslicer")
        .def(py::init<const std::string &, int, int\
              ,int, int, std::vector<std::vector<long>>>())
        .def("getSample", &CSlicer::getSample,py::return_value_policy::take_ownership)
        .def("getNoSamples",&CSlicer::expected_number_of_samples);
        // .def_readwrite("v",&CSlicer::v);
}
