#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <thread>
#include <mutex>
#include <torch/extension.h>
#include <iostream>
#include <util/conqueue.h>

using namespace std;
namespace py = pybind11;



class WrappedData{
  std::vector<int> data;

public:
  torch::Tensor ten;
  WrappedData(std::vector<int> d){
      data = d;
      std::cout << "top:";
      for(int i=0; i <4;i++){
        std::cout << data[i] <<" ";
      }
      std::cout << "check\n";
      auto opts = torch::TensorOptions().dtype(torch::kInt32);
      ten = torch::from_blob(data.data(), {data.size()}, opts);
  }
  ~ WrappedData(){
    // data.clear();
    std::cout << "deleting \n";
  }
};
template class ConQueue<WrappedData *>;

class Producer{
  std::mutex *mtx;
   ConQueue<WrappedData *> *generated_samples[4];
   int num_objects;
   int worker_id;
public:
  Producer(std::mutex *mtx,  ConQueue<WrappedData *> *generated_samples[4],
      int num_objects, int worker_id){
    this->mtx = mtx;
    for(int i=0;i<4;i++){
      this->generated_samples[i] = generated_samples[i];
    }
    this->worker_id = worker_id;
    this->num_objects = num_objects;
  }

  void run(){
    std::vector<int> data;
    for(int i=0;i<100;i ++ ){
      (data).push_back(worker_id);
    }

    for(int n=0; n< num_objects; n++){
      std::unique_lock<std::mutex> lck(*mtx);
        std::cout << "capture global lock " << this->worker_id << "\n";
      for(int j=0;j<4;j++){
        // auto *t = new torch::ones(100);
        generated_samples[j]->push_object(new WrappedData(data));
      }
      std::cout << "release global lock \n";
    }
  }
};

class WorkQueue{

    ConQueue<WrappedData *> *generated_samples[4];
    std::thread **th;
    std::mutex mtx;

public:
  WorkQueue( int num_objects, int num_workers){
    for(int i=0;i<4;i++){
      generated_samples[i] = new ConQueue<WrappedData *>(3);
    }
    th = (std::thread **)malloc(sizeof(std::thread) * num_workers);
    // Start all worker threads on empty queue.
    Producer ** p = (Producer **)malloc(sizeof(Producer) * num_workers);
    for(int i=0;i<num_workers;i++){
      p[i] = new Producer(&mtx, generated_samples,num_objects, i);
      th[i] = new std::thread(&Producer::run, p[i]);
     }

  }

  WrappedData * get_object(int queue_id){
    // std::move is ithis the right call.
    std::cout << "queid "<< queue_id << " "<< generated_samples[queue_id]->get_size() << "\n";
    return std::move(generated_samples[queue_id]->pop_object());
  }


};

PYBIND11_MODULE(cslicer, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    py::class_<WorkQueue>(m,"work")
         .def(py::init<int,int>())
         .def("get_object", &WorkQueue::get_object, py::return_value_policy::take_ownership);
   py::class_<WrappedData>(m,"data")
        .def_readwrite("t",&WrappedData::ten);

    }
