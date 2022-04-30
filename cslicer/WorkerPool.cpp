 #include "WorkerPool.h"
 #include "util/conqueue.h"
 #include <vector>

WorkerPool::WorkerPool(long num_nodes, int num_epochs,
    int minibatch_size, int num_workers, Dataset *dataset,
      std::vector<int>& workload_map,
        std::vector<int>& storage_map[4],
        int batch_size){
  this->num_nodes = num_nodes;
  this->num_epochs = num_epochs;
  this->minibatch_size = minibatch_size;
  this->num_workers = num_workers;
  this->training_nodes = (long *)malloc(sizeof(long)* num_nodes);
  for(long i=0;i<num_nodes;i++){
    this->training_nodes[i] = i;
  }
  this->workload_map = workload_map;
  for(int i=0; i<4; i++){
    this->storage_map[i] = storage_map[i];
  }
  this->num_batches = (num_nodes-1)/minibatch_size + 1;
  this->work_queue = new ConQueue<std::vector<long> *>(10);
  this->generated_samples = new ConQueue<Sample *>(10);
  this->samplers = (CSlicer **)malloc(sizeof(CSlicer *) * no_workers);
}

void WorkerPool::run(){
  std::thread **th = (std::thread **)malloc(sizeof(std::thread) * num_workers);
  // Start all worker threads on empty queue.
  for(int i=0;i<num_workers;i++){
    this->samplers[i] = new CSlicer(dataset, workload_map, storage_map, batch_size,
            *this->generated_samples, *this->work_queue);
    th[i] = new std::thread(CSlicer::run,&this->samplers[i]);
  }
  // vector of training nodes
  for(int epoch = 0;epoch < this->num_epochs; epoch ++ ){
    std::random_shuffle(&this->training_nodes[0], &this->training_nodes[this->num_nodes]);
    for(int j=0; j< this->num_nodes; j = j + this->minibatch_size){
        vector<long> * batch = new vector<long>();
        long end = j + this->minibatch_size;
        if(end > this->num_nodes){
          end = this->num_nodes;
        }
        batch->insert(batch->end(), &this->training_nodes[j], &dataArray[end]);
        this->work_queue(batch);
      }
  }
}
