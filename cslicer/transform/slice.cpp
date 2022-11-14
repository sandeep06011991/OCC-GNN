#include "transform/slice.h"

void Slice::get_edge_policy(vector<long> &in, Block &bl, vector<POLICY> &policy, int layer_id){
    policy.clear();
    vector<gpucost> indegrees;
    vector<gpucost> outdegrees;
    memset(indegrees, 0, sizeof(int))
    for(int i=0;i < in.size(); i++){
        for(int j= offsets[i]:offsets[i+1]){
          if(p_src == p_dest) {
            Policy.pushback(LOCAL)
          }
          int from = this->workload_map[nd2];
          if(layer_id == 2){
            // spdlog::info("Bug related to storage map ordering add a unit test");
            if(this->storage_map[to][nd2] != -1){
              from  = to;
            }

            if(pulled_here){
              from = to;
            }
          }
          // Last layer check for storage in degree
          if(layer_id == 2){
            if(this->storage_map[to][nd2] != -1)continue;
          }
          this->p_indegree[p_src]->increment(nd1);
          this->p_outdegree[p_dest]->increment(nd2);
        }
    }
    for(int i=0;i<in.size();i ++ ){
      for(int j=offsets[i]:offsets[i+1]){
        if(this->p_indegree[i] > this->p_outdegree)policy.pushback(PUSH);
        else policy.pushback(PULL)
      }
    }
    assert(policy.size() == bl.offsets)
}
// FixMe:
// This function is badly built as it does both
// Edge selection, from and to node construction and reordering in the sae function.
// This can be obviously be broken into three stages.
// [Not in this version of the paper] To add redudancy, color all the nodes with bitflags
// And replicate the edge in all partitions which have the bitflag set.
void Slice::slice_layer(vector<long>& in, Block &bl, PartitionedLayer& l, int layer_id, edge_policy){
    // Calculate out_degree and in degree for bipartite graph
    // For pulling node.
    vector<long> partition_edges[4];
    for(int i=0;i<in.size(); i++){
      long nd1 = in[i];
      long in_degree = bl.in_degree[i];
      for(int j= bl.offsets[i] ; j < bl.offsets[i+1]; j++){
        POLICY policy;
        long nd2 = bl.indices[j];
        if(policy == PUSH){
           partition_edges[nd2];
        }else{
           partition_edges[nd1];
        }
      }
      for(int i=0;i<4;i++){
        l.bipartite[i].merge_graph_local(nd1, partition_edges);
        l.bipartite[i].add_degree(nd1, in_degree);
      }else{
        l.bipartite[i].merge_graph_remote(nd1, vector<long> )
      }

        if(nd1 == nd2){
            l.bipartite[to]->add_self_edge(nd1, in_degree);
            // If Attention
            if(this->self_edge){
              l.bipartite[to]->add_edge(nd1, nd1, true);
            }
        }else{

            if(to == from){
              t[to] ++;
              l.bipartite[to]->add_edge(nd1,nd2,true);
            }else{
              t[from] ++;
              l.bipartite[from]->add_edge(nd1,nd2,false);
              l.bipartite[from]->add_to_node(nd1,to);
              l.bipartite[to]->add_from_node(nd1,from);
            }
        }

      }
      for(int i=0;i<4;i++){
        reorder_bipartite_graph
        // preserve ordering of local.
        to reorder
        pull reorder 
      }
    }

    if(layer_id != 2){
      for(int i=0;i<4;i++){
        l.bipartite[i]->reorder(dr);
      }
    }else{
      for(int i=0;i<4;i++){
          l.bipartite[i]->reorder_lastlayer(dr,storage_map[i], gpu_capacity[i]);
      }
    }
    // if (this->deterministic){
    //   layer_consistency(l);
    // }
  }

  void Slice::slice_sample(Sample &s, PartitionedSample &ps){
    // std::cout << "Reached here\n";
    vector<POLICY> edge_policy;
    for(int i= 1; i< s.num_layers + 1;i++){
        PartitionedLayer& l = ps.layers[i-1];
        int layer_id = i-1;
        this->get_edge_policy( *s.block[i], edge_plicy);
        // std::cout << "Attempting to slice ################### \n";
        this->slice_layer(s.block[i-1]->layer_nds,
          (* s.block[i]), l, layer_id);
        // l.debug();
    }
    for(int i=0;i<4;i++){
        ps.refresh_map[i] = ps.layers[2].bipartite[i]->missing_node_ids;
        ps.cache_hit[i] = ps.layers[2].bipartite[i]->cached_node_ids;
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
