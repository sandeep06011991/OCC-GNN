#include "transform/slice.h"

// Get Edge selection.
void Slice::get_edge_policy(vector<long> &in, Block &bl, vector<POLICY> &policy, int layer_id){
    policy.clear();
    vector<gpu> in_degree;
    vector<gpu> out_degree;
    out_degree.size(bl.layer_nds.size());
    in_degree.size(in.size());
    memset(in_degree.data(), 0, sizeof(gpu) * in.size());
    memset(out_degree.data(), 0, sizeof(gpu) * in.size());
    for(int i=0;i < in.size(); i++){
        int p_dest = this->partition_map[in[i]];
        for(int j= offsets[i]; j <offsets[i+1]; j++){
          int nd2 = bl.layer_nds[bl.indices[j]];
          int p_src = this->partition_map[nd2];
          if(layer_id == 2){
            if(this->storage_map[p_dest][nd2] != -1){
              p_src = p_dest;
            }
          }
          if(p_src == p_dest) {
            policy.pushback(LOCAL);
          }else{
            policy.pushback(PUSH);
          }
          int from = this->workload_map[nd2];
          in_degree[i].gpu[p_src] ++;
          out_degree[j].gpu[p_dest] ++;
        }
    }
    for(int i=0;i<in.size();i ++ ){
      int p_dest = this->partition_map[in[i]];
      for(int j= offsets[i]; j <offsets[i+1]; j++){
        int nd2 = bl.layer_nds[bl.indices[j]];
        int p_src = this->partition_map[nd2];
        if(this->p_indegree[i].gpus[p_src] < this->p_outdegree[j].gpus[p_dest] * this->rounds)policy[j] = PULL;
      }
    }
    assert(policy.size() == bl.offsets)
}

// [Not in this version of the paper] To add redudancy, color all the nodes with bitflags
// And replicate the edge in all partitions which have the bitflag set.
//  Push reordering to seperate function.
void Slice::slice_layer(vector<long>& in, Block &bl, PartitionedLayer& l, int layer_id,
          vector<POLICY> &policy){
    // Calculate out_degree and in degree for bipartite graph
    // For pulling node.
    vector<long> partition_edges[4];
    vector<long> pull_nodes[4];
    for(int i=0;i<in.size(); i++){
      for(int i=0;i<4;i++){
        partition_edges[i].clear();
        pull_nodes[i].clear();
      }
      int p_dest = this->partition_map[in[i]]
      for(int j= bl.offsets[i]; j < bl.offsets[i+1]; j++){
        POLICY  p = policy[j];
        long nd_src = bl.layer_nds[bl.indices[j]];
        if(p == PUSH){
          int p_src = this->partition_map[nd_src];
          partition_edges[p_src].push_back(nd_src);
        }
        if(p == LOCAL){
          partition_edges[p_dest].push_back(nd_src);
        }
        if(p == PULL){
          partition_edges[p_dest].push_back(nd_src);
          pull_nodes[p_src].push_back(nd_src);
        }
      }
      long nd_dest = in[i];
      long in_degree = bl.in_degree[i];
      l.bipartite[p_dest].merge_local_graph(partition_edges[p_dest], in_degree, nd_dest);
      for(int src=0;src < 4; src++){
        if(src != p_dest){
          l.bipartite[p_src].merge_remote_graph(partition_edges[src], nd_dest);
          l.bipartite[p_src].merge_pull_nodes(pull_nodes[src]);
        }
      }
    }
  }  

  void Slice::reorder(PartitionedLayer &l){
     for(int i=0;i < 4; i++){
       l.bipartite[i].local_reorder();
     }
     for(int i = 0;i < 4; i++){
       for(int from = 0; from < 4; from ++ ){
         if(i==from)continue;
         dr->order(in_nodes);
         vector<long> incoming_nodes;
         dr->replace(incoming_nodes)
         l.bipartite[i]->from_nodes.push_back(incoming_nodes);
       }
     }
  }

  void Slice::slice_sample(Sample &s, PartitionedSample &ps){
    // std::cout << "Reached here\n";
    vector<POLICY> edge_policy;
    for(int i= 1; i< s.num_layers + 1;i++){
        PartitionedLayer& l = ps.layers[i-1];
        int layer_id = i-1;
        this->get_edge_policy( *s.block[i], edge_plicy);
        // std::cout << "Attempting to slice ################### \n";
        this->partition_edges(s.block[i-1]->layer_nds, \
          (* s.block[i]), l, layer_id);
        this->reorder(l);
        // l.debug();
    }
    for(int i=0;i<4;i++){
        ps.cache_miss_from[i].clear();
        ps.cache_hit_from[i].clear();
        ps.cache_miss_to[i].clear();
        ps.cache_hit_to[i].clear();
        for(int j = 0; j <ps.layers[2].bipartite[i].in_nodes.size(); j++){
          auto nd = ps.layers[2].bipartite[i].in_nodes[j];
          if (this->storage_map[i][nd] != -1){
              ps.cache_hit_from[i].push_back(this->storage_map[i][nd]);
              ps.cache_hit_to[i].push_back(j);
          }else{
            ps.cache_miss_from[i].push_back(nd);
            ps.cache_miss_to[i].push_back(j);
          }
        }
    }
  }



  void Slice::measure_pull_benefits(Sample &s){
      int num_nodes = this->workload_map.size();
      for(int i= 1; i< s.num_layers + 1;i++){
         vector<int> p_indegree[4];
         vector<int> p_outdegree[4];
         vector<int> p_pulled[4];
         vector<int> p_new_indegree[4];
         vector<long>& in_nodes = s.block[i-1]->layer_nds;
         vector<long>& offsets = s.block[i]->offsets;
         vector<long>& indices = s.block[i]->indices;
         for(int i=0;i<4;i++){
           p_indegree[i].resize(num_nodes);
           p_outdegree[i].resize(num_nodes);
           p_pulled[i].resize(num_nodes);
           p_new_indegree[i].resize(num_nodes);
           memset(p_indegree[i].data(), 0, sizeof(int) * num_nodes);
           memset(p_outdegree[i].data(), 0, sizeof(int) * num_nodes);
           memset(p_pulled[i].data(), 0, sizeof(int) * num_nodes);
           memset(p_new_indegree[i].data(),0, sizeof(int) * num_nodes);
        }
         for (int i = 0;i < in_nodes.size(); i++){
           auto nd1 = in_nodes[i];
           int p_dest = this->workload_map[nd1];
           for(int j = offsets[i]; j < offsets[i+1]; j++){
              auto nd2 = indices[j];
              int p_src = this->workload_map[nd2];
              if(p_src == p_dest) continue;
              p_indegree[p_src][nd1] ++ ;
              p_outdegree[p_dest][nd2]++;
           }
         }
        int shuffle_cost  = 0;
        for (int i=0;i<in_nodes.size();i++){
          auto nd1 = in_nodes[i];
          int p_dest = this->workload_map[nd1];
          for(int j=0; j<4; j++){
              if((p_dest != j) && (p_indegree[j][nd1] > 0))shuffle_cost ++;
          }
        }
        std::cout << "Total shuffle cost" << shuffle_cost <<"\n";
        for (int i = 0;i < in_nodes.size(); i++){
          auto nd1 = in_nodes[i];
          int p_dest = this->workload_map[nd1];
          for(int j = offsets[i]; j < offsets[i+1]; j++){
             auto nd2 = indices[j];
             int p_src = this->workload_map[nd2];
             if((p_src != p_dest) && (p_outdegree[p_dest][nd2] > p_indegree[p_src][nd1]/4)){
               p_pulled[p_dest][nd2] = 1;
             }
          }
        }
        for (int i=0;i<in_nodes.size();i++){
          auto nd1 = in_nodes[i];
          int p_dest = this->workload_map[nd1];
          for(int j = offsets[i]; j < offsets[i+1]; j++){
             auto nd2 = indices[j];
             int p_src = this->workload_map[nd2];
             if((p_dest != p_src) && (p_pulled[p_dest][nd2] != 1)) p_new_indegree[p_src][nd1] ++ ;
          }
        }
        int pull = 0;
        for(int i=0;i<num_nodes;i++){
          for(int j=0;j<4;j++){
             if(p_pulled[j][i]==1)pull++;
          }
        }
        shuffle_cost  = 0;
        for (int i=0;i<in_nodes.size();i++){
          auto nd1 = in_nodes[i];
          int p_dest = this->workload_map[nd1];
          for(int j=0; j<4; j++){
              if((p_dest != j) && (p_new_indegree[j][nd1] > 0))shuffle_cost ++;
          }
        }
        memset(p_indegree[0].data(), 0, sizeof(int) * num_nodes);
        for(auto i: indices){
          p_indegree[0][i] = 1;
        }
        int total = 0;
        for(int i=0;i<num_nodes; i++){
          total += p_indegree[0][i];
        }
        std::cout << "Total hybrid cost" << shuffle_cost <<":"<< pull <<":" <<total<<"\n";
      }
  }
