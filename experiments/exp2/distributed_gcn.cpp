#include<iostream>
#include "util/gpu.hh"
#include "util/dist_tensor.hh"
#include "samplers/sample.h"
#include "samplers/full.h"
#include "gnn/dist_sage.hh"
#include "util/timer.h"
#include "dataset.h"
int no_gpus = 4;

int run_experiment(string filename){
  // Dummy dataset
  // std::string filename = "pubmed";
  std::string BIN_DIR = "/home/spolisetty/data/";
  reset_timers();
  active_timer(MOVEMENT_COST);
  active_timer(MOVEMENT_COMPUTE);
  // Dataset * dataset = new Dataset(BIN_DIR + "/pubmed");
  Dataset * dataset = new Dataset(BIN_DIR + filename);
  int no_vertices = dataset->num_nodes;
  int no_edges = dataset->num_edges;
  int fsize = dataset->fsize;
  int no_gpus = 4;
  enable_peer_communication();
  for(int i=0;i<no_vertices * fsize;i++){
    dataset->features[i] = 1;
  }
  // start_timer(MOVEMENT_COST);
  DistTensor * in = new DistTensor(dataset->features, Shape(no_vertices,fsize),
          dataset->partition_map,no_gpus);
  Tensor<float> *in_t1; SageAggr sg1; SageAggr sg2;
  if(verify){
    in_t1 = new Tensor<float>(dataset->features, Shape(no_vertices,fsize),0);
    sg1 = new SageAggr(fsize,0);
    sg2 = new SageAggr(fsize,0);
  }
  // stop_timer(MOVEMENT_COST);
  auto *s = new TwoHopNoSampler(dataset->num_nodes, dataset->num_edges,
             dataset->indptr, dataset->indices, 4096,dataset->features,
              dataset->labels, dataset->fsize);

  DistSageAggr *ll1 = new DistSageAggr(fsize, no_gpus, false);
  DistSageAggr *ll2 = new DistSageAggr(fsize, no_gpus, false);

  int noB = s->number_of_batches();
  noB = min(noB,3);
  for(int bid = 0; bid<noB;bid++){

    std::cout <<  "batch " << bid <<"\n";
    s->get_sample(bid);
    size_t free, total;
    cudaMemGetInfo( &free, &total );
    SampleLayer &l2 = s->sample.l2;
    for(int i=0;i<l2.indices.size();i++){
      l2.indices[i] = l2.nd2[l2.indices[i]];
    }
    // std::cout << l2.nd2.size() << " " << l2.indices.size() <<"\n";
    // assert(l2.nd2.size() == l2.indices.size());

    ll1->forward(l2.indptr,l2.indices, *in,l2.indptr.size()-1, no_vertices);
    SampleLayer &l1 = s->sample.l1;
    ll2->forward(l1.indptr,l1.indices, *ll1->out_feat, l1.indptr.size()-1, ll1->out_feat->s.dim1);
    sync_all_gpus();
    if(verify){
      e
    }

  }
  print_timer();

}

int main(int argc, char *argv[]){
  std::string filename = (argv[1]);
  run_experiment(filename);
}
