#include "transform/slice.h"
#include <spdlog/spdlog.h>
// FixMe:
// This function is badly built as it does both
// Edge selection, from and to node construction and reordering in the sae function.
// This can be obviously be broken into three stages.
// [Not in this version of the paper] To add redudancy, color all the nodes with bitflags
// And replicate the edge in all partitions which have the bitflag set.
void Slice::slice_layer(vector<long>& in, Block &bl, PartitionedLayer& l, int layer_id){
    int reduction = 0;
    for(int i=0;i<in.size(); i++){
      long nd1 = in[i];
      long in_degree = bl.in_degree[i];
      int to = this->workload_map[nd1];
      int t[4];
      for(int ii = 0;ii<4; ii ++ ){
        t[ii] = 0;
      }
      int a[4];
      for(int i=0;i<4;i++)a[i] == 0;
      for(int j=bl.offsets[i];j <bl.offsets[i+1]; j++){
        long nd2 = bl.indices[j];
        int from = this->workload_map[nd2];
        a[from] ++;
      }
      for(int i=0;i<4;i++){
        if((a[i] == 1) || (a[i]== 2)){
          reduction ++;
        }
      }
      for(int j= bl.offsets[i] ; j < bl.offsets[i+1]; j++){
        long nd2 = bl.indices[j];
        if(nd1 == nd2){
            l.bipartite[to]->add_self_edge(nd1, in_degree);
            // If Attention
            if(this->self_edge){
              l.bipartite[to]->add_edge(nd1, nd1, true);
            }
        }else{
            int from = this->workload_map[nd2];
            if(layer_id == 2){
              // spdlog::info("Bug related to storage map ordering add a unit test");
              if(this->storage_map[to][nd2] != -1){
                from  = to;
              }
            }
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
    if(reduction > 0){
      int shuffle = 0;
      for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
          shuffle += l.bipartite[i]->from_ids[j].size();
        }
      }
      std::cout << "shuffle" << shuffle << " " << reduction <<"\n";
    }
    // if (this->deterministic){
    //   layer_consistency(l);
    // }
  }

  void Slice::slice_sample(Sample &s, PartitionedSample &ps){
    // std::cout << "Reached here\n";
    for(int i= 1; i< s.num_layers + 1;i++){
        PartitionedLayer& l = ps.layers[i-1];
        int layer_id = i-1;
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



  void Slice::measure_pull_benefits(Sample &s, int rounds){
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

        std::cout << "Total hybrid cost" << shuffle_cost <<":"<< pull <<"\n";
      }
  }
