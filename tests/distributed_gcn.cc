#include<iostream>
#include<dist_tensor.hh>
#include<tensor.hh>

int main(){
// Dummy dataset
  int no_vertices = 10;
  int no_edges = 10;
  int src[] = {};
  int dest[] = {};
  int fsize = 32;
  float *f_cpu = (float *)malloc(fsize * no_vertices);
// Sample
  int sample_src[] = {1};
  int sample_dest[] = {2,3,4,5};
// GPU data
  int no_gpus = 2;
  int device_ids[] = {0,1};


  Shape s1(10,10);
  Shape s2(10,20);
  std::cout << (s1==s2) <<"\n";

// Commit 1.
// Partition f_cpu into a multi gpu container.
// Either takes in partition map
// or randomly partition the features.
  // DistributedTensor * dt = new DistributedTensor();
  // DistributedTensor * in = new DistributedTensor();
  // run_function_distributed_aggregation(out, in, global_sample_map);
  std::cout << "Hello word\n";
}
