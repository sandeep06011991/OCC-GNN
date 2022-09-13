#include "transform/slice.h"

// FixMe:
// This function is badly built as it does both
// Edge selection, from and to node construction and reordering in the sae function.
// This can be obviously be broken into three stages.
// To add redudancy, color all the nodes with bitflags
// And replicate the edge in all partitions which have the bitflag set. 
void Slice::slice_layer(vector<long>& in, Block &bl, PartitionedLayer& l, int layer_id){
    for(int i=0;i<in.size(); i++){
      long nd1 = in[i];
      long in_degree = bl.in_degree[i];
      int to = this->workload_map[nd1];
      int t[4];
      for(int ii = 0;ii<4; ii ++ ){
        t[ii] = 0;
      }
      for(int j= bl.offsets[i] ; j < bl.offsets[i+1]; j++){
        long nd2 = bl.indices[j];
        if(nd1 == nd2){
            l.bipartite[to]->add_self_edge(nd1, in_degree);
        }else{
            int from = this->workload_map[nd2];
            if(layer_id == 2){
              if(this->storage_map[to][nd2] != 0){
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
    // if (this->deterministic){
    //   layer_consistency(l);
    // }
  }

  void Slice::slice_sample(Sample &s, PartitionedSample &ps){
    for(int i= 1; i< s.num_layers + 1;i++){
        PartitionedLayer& l = ps.layers[i-1];
        int layer_id = i-1;
        this->slice_layer(s.block[i-1]->layer_nds,
          (* s.block[i]), l, layer_id);
    }
  }
