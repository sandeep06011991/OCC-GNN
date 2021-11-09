#include<iostream>
#include  "util/tensor.hh"
#include "util/dist_tensor.hh"
// #include "gnn/dist_sage.hh"
// #include<tensor.hh>

int main(){
// Dummy dataset
  int no_vertices = 6;
  int no_edges = 6;

  int fsize = 8;
  float *f_cpu = (float *)malloc(sizeof(float) * fsize * no_vertices);
  for(int i=0;  i< (fsize * no_vertices); i++){
      f_cpu[i] = i/fsize;
  }
// Sample
  int sample_src[] = {1};
  int sample_dest[] = {2,3,4,5};
// GPU data
  int no_gpus = 2;
  // int device_ids[] = {0,1};

  // Tensor<float> *t;
  // Tensor<float> *t = new Tensor<float>(Shape(1,2),1);
  int ordering[] = {0,1,2,3,0,1};
  DistTensor * in = new DistTensor(f_cpu, Shape(no_vertices,fsize), ordering);
  in->debugTensor();
  // DistributedTensor *in = new DistributedTensor(f_cpu, shape, ordering);
  // DistributedGCNLayer *ll = new DistributedGCNLayer(fsize);
  // DistributedTensor &out = l1->forward(*in, sample_src, sample_dest);


// // Commit 1.
//     Allocate and move data for distributed in
// // Commit 2.
//     Allocated and move data for distributed out. [src = dist]
// // Commit 3.
//     Launch local kernels for aggregation.
// // Commit 4.
//     Move and reduce data.
// // Commit 5.
//     Debug.

// Commit 1.
// Partition f_cpu into a multi gpu container.
// Either takes in partition map
// or randomly partition the features.
  // DistributedTensor * dt = new DistributedTensor();
  // DistributedTensor * in = new DistributedTensor();
  // run_function_distributed_aggregation(out, in, global_sample_map);
  std::cout << "Hello word\n";
}
