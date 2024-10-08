#include<iostream>
#include "util/gpu.hh"
// #include  "util/tensor.hh"
#include "util/dist_tensor.hh"
#include "samplers/sample.h"
#include "gnn/dist_sage.hh"
#include "util/timer.h"
// #include "gnn/dist_sage.hh"
// Toy graph running.

int no_gpus = 2;

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
  int sample_src[] = {1,1,1,2,2,2};
  int sample_dest[] = {0,1,2,3,4,5};
  SampleLayer ss1;
  for(int i=0;i<sizeof(sample_src)/sizeof(int);i++){
    ss1.nd1.push_back(sample_src[i]);
    ss1.nd2.push_back(sample_dest[i]);
    ss1.edges.push_back(std::make_pair(sample_src[i],sample_dest[i]));
  }
  ss1.remove_duplicates();
  ss1.create_csr();
//
// // GPU data
  int no_gpus = 2;
  enable_peer_communication();
  int ordering_in[] = {0,1,0,1,0,1};
  int ordering_out[] = {0,1};
  active_timer(TIME1);
  // start_timer(TIME1);
  DistTensor * in = new DistTensor(f_cpu, Shape(no_vertices,fsize), ordering_in,no_gpus);
  // stop_timer(TIME1);
  // print_timer();
  // in->debugTensor();
  DistSageAggr *ll = new DistSageAggr(fsize, no_gpus);
//   // in->debugTensor();
  for(int i=0;i<2;i++){
    ll->forward(ss1.indptr,ss1.indices, *in, ss1.indptr.size()-1,ss1.indices.size()-1);
  }
  ll->out_feat->debugTensor();
//   // DistributedTensor *in = new DistributedTensor(f_cpu, shape, ordering);
//   // DistributedGCNLayer *ll = new DistributedGCNLayer(fsize);
//   // DistributedTensor &out = l1->forward(*in, sample_src, sample_dest);
//
//
// // Commit 1.
// Partition f_cpu into a multi gpu container.
// Either takes in partition map
// or randomly partition the features.
  // DistributedTensor * dt = new DistributedTensor();
  // DistributedTensor * in = new DistributedTensor();
  // run_function_distributed_aggregation(out, in, global_sample_map);
  std::cout << "Hello word\n";
}
