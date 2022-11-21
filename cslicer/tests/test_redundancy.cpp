#include "test.h"
#include "graph/sample.h"
#include "graph/sliced_sample.h"
#include "transform/slice.h"
#include <iostream>

redundant get_red_stats(PartitionedSample &ps){
  redundant r;
  for(int i =  ps.num_layers-1  ; i>=0; i--){
    PartitionedLayer &layer = ps.layers[i];
    // PULL
    for(int j=0;j < 4;j ++){
      r.total_computation += layer.bipartite[j]->indices_L.size() + \
              layer.bipartite[j]->indices_R.size();

      r.total_communication += layer.bipartite[j]->num_out_remote + \
                  layer.bipartite[j]->num_in_nodes_pulled;
    }
  }

  for(int i=0;i<4;i++){
    r.total_computation += ps.cache_miss_to[i].size();
  }
  return r;

}

void test_reduction_communication_computation(Sample &s,std::vector<int> workload_map,
          std::vector<int> storage[4], std::vector<int> storage_map[4], int rounds ){
      vector<int> ** layer_color = (vector<int> **)malloc(4 * sizeof(vector<int> *));
      for(int i=0;i<4;i++){
        layer_color[i] = new vector<int>();
        assert(storage_map[i].size() == workload_map.size());
        for(long j=0;j<workload_map.size();j++){
              layer_color[i]->push_back(0);
        }
      }
      bool self_edge = false;
      bool pull_optim= true;
      Slice *s1 =   new Slice(workload_map, storage, self_edge, rounds, pull_optim);
      std::cout << s.num_layers <<" ";
      PartitionedSample ps1(s.num_layers);
      s1->slice_sample(s, ps1);

      redundant r1 = print_statistics(s, layer_color,workload_map.size(), workload_map, storage_map);
      redundant r2 = get_red_stats(ps1) ;
      std::cout << "Baseline\n";
      r1.debug();
      std::cout << "My approach\n";
      r2.debug();
}
