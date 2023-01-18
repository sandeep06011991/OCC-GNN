#include "transform/slice.h"
#include <cstring>
#include "tests/test.h"


__global__ get_offsets(long*  partition_map, long * indptr, long *indices,\
        void * indptr_map, void * indices_map){
    int tid = blockId.x *blockDim.x + threadId.x;
    while(tid < size){
      long nd1 = layer_nds[tid];
      long nbs = offsets[tid+1] - offset[td];
      for(int n = 0; n<nbs; n ++ ){
        nd2 = layer_nds[indptr[nd1]];
        if(p_map[nd2] == partition_id){

        }
        for(int i=0;i < indptr; i++){

        }
      }
      tid += (blockDim.x * gridDim.x);
    }
}


void Slice::reorder(PartitionedLayer &l){
     for(int i=0;i < this->num_gpus; i++){
   	   l.bipartite[i]->reorder_local(dr, this->num_gpus);
     }
     // Handle remote destination nodes
     for(int to = 0; to < this->num_gpus; to ++){
       dr->clear();
       dr->order_and_remove_duplicates(l.bipartite[to]->out_nodes_local);
       for(int from = 0; from < this->num_gpus; from++){
	        if(from == to) continue;
         int start = l.bipartite[from]->to_offsets[to];
         int end = l.bipartite[from]->to_offsets[to + 1];
         l.bipartite[to]->from_ids[from].clear();
         vector<long> &t = l.bipartite[to]->from_ids[from];
      	 vector<long> &f = l.bipartite[from]->out_nodes_remote;
      	 t.insert(t.end(), f.begin() + start, f.begin() + end );
      	 dr->replace(t);
       }
     }
     // Think on paper what I am trying to do here.
     for(int pull_from = 0;pull_from < this->num_gpus; pull_from++){
       dr->clear();
       dr->order_and_remove_duplicates(l.bipartite[pull_from]->in_nodes);
       for(int pull_to = 0; pull_to < this->num_gpus; pull_to ++ ){
         if(pull_from == pull_to)continue;
         int start = l.bipartite[pull_to]->pull_from_offsets[pull_from];
         int end = l.bipartite[pull_to]->pull_from_offsets[pull_from + 1];
         vector<long> &f = l.bipartite[pull_from]->push_to_ids[pull_to];
      	 vector<long> &t = l.bipartite[pull_to]->pulled_in_nodes;
         assert((end-start) <= t.size());
         f.clear();
      	 f.insert(f.end(), t.begin() + start, t.begin() + end);
         dr->replace(f);
       }
     }
  }

void Slice::slice_layer(thrust::device_vector<long> &layer_nds,
      Block &bs, Bipartite *bs, int partition_id){
    thrust::device_vector<void *> indptrs(4);
    thrust::device_vector<void *> indices(4);
    for(int i=0;i<4;i++){
      bs->indptr[i].size(layer_nds);
      indptrs[i] = bs->indptr[i];
      // indices[i] = bs->indices[i];
    }
    // Get global Indptr Offsets
    get_global_order_offsets();

    // Inclusive Scan
    for(int i=0; i<4;i ++){
      thrust::inclusive_scan(offsets.end(), offsets.begin(), offsets.end());
      indices.resize(offsets[size-1]);

    }
    // Put in indices
    for(int i=0;i <4;i++){

    }
    // All Local graphs are populated
    // at the end of this step.
    for(int src=0;src < this->num_gpus; src++){
        // Its not actually src but destination for remote nodes in this line.

        // Contains local and remote nodes if src == p_dest local, else remote
        if((src == p_dest) ||  (partition_edges[src].size() != 0)){
            l.bipartite[src]->merge_graph(partition_edges[src], nd_dest, p_dest);
          }
        // Local edges, but p_dest must pull node from p_src
        if(pull_nodes[src].size() != 0) l.bipartite[p_dest]->merge_pull_nodes(pull_nodes[src], src);
        // local in nodes for every partition, seperate from self nodes
      if(local_in_nodes[src].size()!=0)l.bipartite[src]->merge_local_in_nodes(local_in_nodes[src]);
    }
}


void Slice::slice_sample(Sample &s, PartitionedSample &ps){

    for(int i= 1; i< s.num_layers + 1;i++){
	   // std::cout << "Sliceed sample \n";
	    PartitionedLayer& l = ps.layers[i-1];
        int layer_id = i-1;
        this->slice_layer(s.block[i-1]->layer_nds, \
          (* s.block[i]), l, layer_id, edge_policy);
	     this->reorder(l);
    }

    for(int i=0;i<this->num_gpus;i++){
        ps.cache_miss_from[i].clear();
        ps.cache_hit_from[i].clear();
        ps.cache_miss_to[i].clear();
        ps.cache_hit_to[i].clear();
        ps.last_layer_nodes[i].clear();
          // ps.layers[s.num_layers-1].bipartite[i]->debug();
        for(int j = 0; j <ps.layers[s.num_layers- 1].bipartite[i]->in_nodes.size(); j++){
          auto nd = ps.layers[s.num_layers-1].bipartite[i]->in_nodes[j];
          if (this->storage_map[i][nd] != -1){
              ps.cache_hit_from[i].push_back(this->storage_map[i][nd]);
              ps.cache_hit_to[i].push_back(j);
          }else{
            ps.cache_miss_from[i].push_back(nd);
            ps.cache_miss_to[i].push_back(j);
          }
        }
        vector<long> &last_layer = ps.layers[0].bipartite[i]->out_nodes_local;
        ps.last_layer_nodes[i].insert(ps.last_layer_nodes[i].end(), last_layer.begin(), last_layer.end());
    }
    edge_policy.clear();
	if(check){
// std::cout << "Skipping Cross checking if there are actually 0 remote subgraphs\n";
	//s.check_remote(this->workload_map);
	//test_sample_partition_consistency_gat(s,ps, \
	//   this->storage, this->gpu_capacity, this->workload_map.size());
    }
  }
