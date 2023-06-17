#include "transform/slice.h"
#include "graph/sliced_sample.h"

int measure_communication(int rounds_per_layer, PartitionedSample &ps){
  // refresh storage map with local_ids.
  int total_comm = 0;
  int total_pulled = 0;
  int total_shuffled = 0;
  for(int i =  ps.num_layers-1  ; i>=0; i--){
    PartitionedLayer &layer = ps.layers[i];
    // PULL
    int new_pull = 0;
    for(int j=0;j < 4;j ++){
      BiPartite *bp = layer.bipartite[j];
      total_pulled += bp->num_in_nodes_pulled;
      total_shuffled += bp->num_out_remote;
    }
  }
  std::cout << total_pulled << " " << total_shuffled <<"\n";
  return total_pulled + total_shuffled * rounds_per_layer;
}

void  test_pull_benefits(Sample &s, std::vector<int> workload_map, \
    std::vector<int> storage[4], int rounds){
    bool self_edge = false;
    bool pull_optim= true;
    int num_gpus = 4;
    Slice *s1 =   new Slice(workload_map, storage, self_edge, rounds, pull_optim,num_gpus);
    PartitionedSample ps1(s.num_layers, num_gpus);
    s1->slice_sample(s, ps1);
    pull_optim = false;
    std::cout << pull_optim <<"xxxxxxxxxxxxxxxxxxxxxxxx\n";
    Slice *s2 =   new Slice(workload_map, storage, self_edge, rounds, pull_optim, num_gpus);
    PartitionedSample ps2(s.num_layers, num_gpus);
    ps2.clear();
    s2->slice_sample(s,ps2);
    int pull = measure_communication(rounds,ps1);
    int push = measure_communication(rounds, ps2);
    std::cout << "Savings" << pull <<":"<<push <<"\n";
    assert(pull <= push);
}
