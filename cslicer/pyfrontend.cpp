#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <string>
#include "object.h"
#include "pybipartite.h"

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

class CSlicer{
    std::string graph_name;
    int queue_size;
    int no_worker_threads;
    int number_of_epochs;
    int samples_per_epoch;
    int minibatch_size;
    int samples_generated = 0;
public:
    py::list v;
    std::thread *th;
    CSlicer(const std::string &name, int queue_size, int no_worker_threads \
        , int number_of_epochs, int samples_per_epoch, int minibatch_size){
        this->name = name;
        this->queue_size = queue_size;
        this->no_worker_threads = no_worker_threads;
        this->number_of_epochs = number_of_epochs;
        this->samples_per_epoch = samples_per_epoch;
        this->minibatch_size = minibatch_size;
        this->queue = new ConQueue<PySample *>(queue_size);
        this->pool = new WorkerPool(no_worker_threads, number_of_epochs,
            this->queue, minibatch_size);
        th = new thread(WorkerPool::run, this->pool);
    }
    // Fix serialization issue.

    PySample * getSample(){
      // Sample *s = Sample::get_dummy_sample();
      // return new PySample(s);
      this->pool->generated_samples->pop_object();
    }
};
// Pet::Pet(const std::string &name) : name(name) { }
// void Pet::setName(const std::string &name_) { name = name_; }
// const std::string &Pet::getName() const { return name; }

py::list testlist(py::list l) {
    // l.attr("pop")();
    // std::cout << "List has length " << l.size() << std::endl;
    // for (py::handle obj : l) {  // iterators!
    //     std::cout << "  - " << obj.attr("__str__")().cast<std::string>() << std::endl;
    // }
    std::vector<int> test_vec{1, 2, 3, 4, 5};
    py::list test_list3 = py::cast(test_vec);
    l.append(10);  // automatic casting (through templating)!
    return test_list3;
}

PYBIND11_MODULE(cslicer, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("test_list", &testlist, "List testing ");
    m.def("add", &add, "A function that adds two numbers");
    py::class_<Pet>(m, "Pet")
        .def(py::init<const std::string &>())
        .def("setName", &Pet::setName)
        .def("getName", &Pet::getName);
    py::class_<PySample>(m,"sample")
        .def_readwrite("layers",&PySample::layers);

    py::class_<PyBipartite>(m,"bipatite")
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
        .def(py::init<const std::string &,int, int>())
        .def("getSample", &CSlicer::getSample,py::return_value_policy::take_ownership)
        .def_readwrite("v",&CSlicer::v);
}
