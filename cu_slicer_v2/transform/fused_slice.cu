
#include "slice.h"
#include <cstring>
#include "../graph/bipartite.h"
#include "../util/cub.h"
#include "../util/cuda_utils.h"
#include "../util/types.h"
#include "../graph/order_book.h"
using namespace cuslicer;

void FusedSlicer::slice_layer(device_vector<NDTYPE> &layer_nds,
      Block &bs, PartitionedLayer &ps, bool last_layer){
        
      if(!last_layer){
        for(int i=0; i < this->num_gpus; i++){
            BiPartite &bp = *ps.bipartite[i];
            bp.gpu_id = i;
            bp.num_gpus = this->num_gpus;
            bp.in_nodes_local = bs.layer_nds;
            bp.out_nodes_local = layer_nds;
            bp.indptr_L = bs.offsets;
            bp.indices_L = bs.indices;
            bp.out_degree_local = bs.in_degree;
            bp.indptr_R.clear();
            bp.indices_R.clear();
            bp.pulled_in_nodes.clear();
            bp.pull_from_offsets[0] = 0;
            bp.to_offsets[0] = 0;
            for(int j = 0; j < this->num_gpus; j ++ ){
                bp.pull_from_offsets[j+1] = 0;
                bp.to_offsets[j+1] = 0;
                bp.pull_from_ids_[j].clear();
                bp.push_to_ids_[j].clear();
            }
         }
      }else{
        PullSlicer::slice_layer(layer_nds, bs, ps, last_layer);
        // call pull slicer code 
      }
    }