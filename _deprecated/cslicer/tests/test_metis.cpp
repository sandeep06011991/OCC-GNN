#include<iostream>
#include "graph/dataset.h"
#include "graph/sample.h"
#include "util/environment.h"
#include "samplers/samplers.h"
#include "transform/slice.h"
int main(){
  // Test1: Read graph datastructure.
  std::string graph_name = "ogbn-products";
  std::string file = get_dataset_dir() + graph_name;
  std::shared_ptr<Dataset> dataset = std::make_shared<Dataset>(file);

  // Test2: Construct simple k-hop neighbourhood sample.
  // Sample datastructure.
  Sample *s  = new Sample(3);
  int fanout = 15;
  int batch_size = 4096;
  // Add code with batch size. for the whole graph.
  std::vector<long> training_nodes;
  for(int i=0;i<dataset->num_nodes; i = i + batch_size){
    training_nodes.clear();
    for(int j = i; j < (i + batch_size) && j < dataset->num_nodes; j++){
      training_nodes.push_back(j);
    }
    NeighbourSampler *ns  =  new NeighbourSampler(dataset, fanout);
    ns->sample(training_nodes,(*s));
    check_allocation_for_optimality((*s), dataset->partition_map, dataset->num_nodes);

  }


 std::cout << "Populate initial input \n";
}
