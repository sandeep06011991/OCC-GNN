#include<iostream>
#include "util/gpu.hh"
#include "util/dist_tensor.hh"
#include "samplers/sample.h"
// #include "samplers/full.h"
#include "samplers/neigh_sample.h"
#include "gnn/dist_sage.hh"
#include "util/timer.h"
#include "dataset.h"
int no_gpus = 4;

int run_experiment(string filename,bool isexternal, bool israndom,int hops){
  // Dummy dataset
  // std::string filename = "pubmed";
  std::string BIN_DIR = "/home/spolisetty/data/";
  reset_timers();
  active_timer(MOVEMENT_COST);
  active_timer(MOVEMENT_COMPUTE1);
  active_timer(MOVEMENT_COMPUTE2);
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
  // stop_timer(MOVEMENT_COST);
  auto *s = new TwoHopNeighSampler(dataset->num_nodes, dataset->num_edges,
             dataset->indptr, dataset->indices, 4096,dataset->features,
              dataset->labels, dataset->fsize);

  DistSageAggr *ll1 = new DistSageAggr(fsize, no_gpus, israndom, isexternal);
  DistSageAggr *ll2 = new DistSageAggr(fsize, no_gpus, israndom, isexternal);
  bool verify = false;
  Tensor<float> *v_in;
  if(verify){
    v_in = new Tensor<float>(dataset->features,Shape(no_vertices,fsize),0);
  }
  int noB = s->number_of_batches();
  // noB =8;
  std::cout << "contains a total of no batch" << noB << "\n";
  for(int bid = 0; bid<noB;bid++){
    // std::cout <<  "batch " << bid <<"\n";
    s->get_sample(bid);
    size_t free_m, total;
    cudaMemGetInfo( &free_m, &total );
    SampleLayer &l2 = s->sample.l2;
    for(int i=0;i<l2.indices.size();i++){
      l2.indices[i] = l2.nd2[l2.indices[i]];
    }
    //
    SampleLayer &l1 = s->sample.l1;
    int * partition_map2 = (int *)malloc(sizeof(int) * l2.nd1.size());
    for(int i=0;i<l2.nd1.size();i++){
      partition_map2[i] = dataset->partition_map[l2.nd1[i]];
    }
    int * partition_map1 = (int *)malloc(sizeof(int) * l1.nd1.size());
    for(int i=0;i<l1.nd1.size();i++){
      partition_map1[i] = dataset->partition_map[l1.nd1[i]];
    }
    // std::cout << l2.nd2.size() << " " << l2.indices.size() <<"\n";
    // assert(l2.nd2.size() == l2.indices.size());

    ll1->forward(l2.indptr,l2.indices, *in,l2.indptr.size()-1, no_vertices, partition_map2);
    // SampleLayer &l1 = s->sample.l1;
    ll2->forward(l1.indptr,l1.indices, *ll1->out_feat, l1.indptr.size()-1, ll1->out_feat->s.dim1, partition_map1);
    sync_all_gpus();
    free(partition_map1);
    free(partition_map2);

    if(verify && bid
      %5==0){
        ll2->out_feat->viewTensor();
        SageAggr * sg1 = new SageAggr(fsize,0);
        SageAggr * sg2 = new SageAggr(fsize,0);
        Tensor<int> *indptr2 = new Tensor<int>(l2.indptr.data(),Shape(l2.indptr.size(),1),0);
        Tensor<int> *indices2 = new Tensor<int>(l2.indices.data(),Shape(l2.indices.size(),1),0);
        Tensor<int> *indptr1 = new Tensor<int>(l1.indptr.data(),Shape(l1.indptr.size(),1),0);
        Tensor<int> *indices1 = new Tensor<int>(l1.indices.data(),Shape(l1.indices.size(),1),0);
        auto i1 = sg1->forward(*indptr2, *indices2, *v_in, l2.indptr.size()-1, no_vertices);
        auto i2 = sg2->forward(*indptr1, *indices1, *i1, l1.indptr.size()-1, l2.indptr.size()-1);
        i2->viewTensor();
    }
  }
  print_timer();

}

int main(int argc, char *argv[]){
  std::string filename = (argv[1]);
  // run_experiment(filename);

  std::string partition = (argv[2]);
  // int hops = std::stoi(argv[3]);
  // assert(hops <4);
  bool  found = false;
  bool isexternal;
  bool israndom;
  if(partition == "random"){
    found = true;
    isexternal = false;
    israndom = true;
  }
  if(partition == "metis"){
    found = true;
    isexternal = true;
    israndom = false;
  }
  if(partition == "optimum"){
    found = true;
    isexternal = false;
    israndom = false;
  }
  assert(found);
  run_experiment(filename,isexternal,israndom,hops);
}
