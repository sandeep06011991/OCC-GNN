#include "transform/slice.h"
#include <cstring>
// #include "tests/test.h"
// Get Edge selection.
// void Slice::get_edge_policy(vector<long> &in, Block &bl, vector<POLICY> &policy,
//     int layer_id, int no_layers){
//     policy.clear();
//     vector<gpu> in_degree;
//     vector<gpu> out_degree;
//     gpu zero;
//     for(int i = 0; i < this->num_gpus; i++){
//        zero.cost[i] = 0;
//     };
//     for(int i=0; i < bl.layer_nds.size(); i++){
//       out_degree.push_back(zero);
//     }
//     for(int i=0;i < in.size(); i++){
//       in_degree.push_back(zero);
//     }
//     for(int i=0;i < in.size(); i++){
//         int p_dest = this->workload_map[in[i]];
//         for(int j= bl.offsets[i]; j < bl.offsets[i+1]; j++){
//           int nd2 = bl.layer_nds[bl.indices[j]];
//           int p_src = this->workload_map[nd2];
//           if(layer_id == (no_layers - 1)){
//             if(this->storage_map[p_dest][nd2] != -1){
//               p_src = p_dest;
//             }
//           }
//           if(p_src == p_dest) {
//             policy.push_back(LOCAL);
// 	    //std::cout << "local" << in[i] << " " << nd2 <<"\n";
// 	    continue;
//           }else{
//             policy.push_back(PUSH);
//           }
//           int from = this->workload_map[nd2];
//           in_degree[i].cost[p_src] ++;
//           assert(bl.indices[j] < out_degree.size());
//           out_degree[bl.indices[j]].cost[p_dest] ++;
//         }
//     }
//     assert(bl.indices.size()== policy.size());
//     if(this->pull_optimization && (layer_id ==(no_layers-1))){
//       for(int i=0;i<in.size();i ++ ){
//         int p_dest = this->workload_map[in[i]];
//         for(int j= bl.offsets[i]; j <bl.offsets[i+1]; j++){
//           int nd2 = bl.layer_nds[bl.indices[j]];
//           int p_src = this->workload_map[nd2];
//           if(layer_id == (no_layers - 1)){
//             if(this->storage_map[p_dest][nd2] != -1){
//               p_src = p_dest;
//             }
//           }
//           if(policy[j] != PUSH)continue;
// 	  policy[j] = PULL;
// 	  //assert(policy[j]==PUSH);
//           //if(in_degree[i].cost[p_src] < out_degree[bl.indices[j]].cost[p_dest] * this->rounds){
//             //policy[j] = PULL;
//          // }
//         }
//       }
//     }
// }
//
// // [Not in this version of the paper] To add redudancy, color all the nodes with bitflags
// // And replicate the edge in all partitions which have the bitflag set.
// //  Push reordering to seperate function.
// void Slice::slice_layer(vector<long>& in, Block &bl, PartitionedLayer& l, int layer_id,
//           vector<POLICY> &policy){
//     // Calculate out_degree and in degree for bipartite graph
//     // For pulling node.
//     vector<long> partition_edges[this->num_gpus];
//     vector<long> pull_nodes[this->num_gpus];
//     vector<long> local_in_nodes[this->num_gpus];
//     // std::cout << "Partition count " << this->num_gpus <<"\n";
//     for(int i=0;i<in.size(); i++){
//
//       for(int i=0;i<this->num_gpus;i++){
//         partition_edges[i].clear();
//         pull_nodes[i].clear();
//         local_in_nodes[i].clear();
//       }
//       int p_dest = this->workload_map[in[i]];
//       long nd_dest = in[i];
//       long in_degree = bl.in_degree[i];
//       for(int j= bl.offsets[i]; j < bl.offsets[i+1]; j++){
//         POLICY  p = policy[j];
//         long nd_src = bl.layer_nds[bl.indices[j]];
//         int p_src = this->workload_map[nd_src];
//         if(p == PUSH){
//           partition_edges[p_src].push_back(nd_src);
//           local_in_nodes[p_src].push_back(nd_src);
//         }
//         if(p == LOCAL){
//             partition_edges[p_dest].push_back(nd_src);
//             local_in_nodes[p_dest].push_back(nd_src);
//             }
//         if(p == PULL){
//           assert(p_src != p_dest);
//           partition_edges[p_dest].push_back(nd_src);
//           local_in_nodes[p_src].push_back(nd_src);
//           pull_nodes[p_src].push_back(nd_src);
//         }
//       }
//
//       l.bipartite[p_dest]->add_local_out_node(nd_dest, in_degree);
//
//       for(int src=0;src < this->num_gpus; src++){
//           // Its not actually src but destination for remote nodes in this line.
//
//           // Contains local and remote nodes if src == p_dest local, else remote
//           if((src == p_dest) ||  (partition_edges[src].size() != 0)){
//               l.bipartite[src]->merge_graph(partition_edges[src], nd_dest, p_dest);
//             }
//           // Local edges, but p_dest must pull node from p_src
//           if(pull_nodes[src].size() != 0) l.bipartite[p_dest]->merge_pull_nodes(pull_nodes[src], src);
//           // local in nodes for every partition, seperate from self nodes
//     	  if(local_in_nodes[src].size()!=0)l.bipartite[src]->merge_local_in_nodes(local_in_nodes[src]);
//       }
//     }
//   }
//
//   void Slice::reorder(PartitionedLayer &l){
//      for(int i=0;i < this->num_gpus; i++){
//    	l.bipartite[i]->reorder_local(dr, this->num_gpus);
//      }
//      // Handle remote destination nodes
//      for(int to = 0; to < this->num_gpus; to ++){
//        dr->clear();
//        dr->order_and_remove_duplicates(l.bipartite[to]->out_nodes_local);
//        for(int from = 0; from < this->num_gpus; from++){
// 	        if(from == to) continue;
//          int start = l.bipartite[from]->to_offsets[to];
//          int end = l.bipartite[from]->to_offsets[to + 1];
//          l.bipartite[to]->from_ids[from].clear();
//          vector<long> &t = l.bipartite[to]->from_ids[from];
//       	 vector<long> &f = l.bipartite[from]->out_nodes_remote;
//       	 t.insert(t.end(), f.begin() + start, f.begin() + end );
//       	 dr->replace(t);
//        }
//      }
//      // Think on paper what I am trying to do here.
//      for(int pull_from = 0;pull_from < this->num_gpus; pull_from++){
//        dr->clear();
//        dr->order_and_remove_duplicates(l.bipartite[pull_from]->in_nodes);
//        for(int pull_to = 0; pull_to < this->num_gpus; pull_to ++ ){
//          if(pull_from == pull_to)continue;
//          int start = l.bipartite[pull_to]->pull_from_offsets[pull_from];
//          int end = l.bipartite[pull_to]->pull_from_offsets[pull_from + 1];
//          vector<long> &f = l.bipartite[pull_from]->push_to_ids[pull_to];
//       	 vector<long> &t = l.bipartite[pull_to]->pulled_in_nodes;
//          assert((end-start) <= t.size());
//          f.clear();
//       	 f.insert(f.end(), t.begin() + start, t.begin() + end);
//          dr->replace(f);
//        }
//      }
//   }
//
//   void Slice::slice_sample(Sample &s, PartitionedSample &ps){
//
//     vector<POLICY> edge_policy;
//     // s.debug();
//     bool check = false;
//     for(int i= 1; i< s.num_layers + 1;i++){
// 	   // std::cout << "Sliceed sample \n";
// 	    PartitionedLayer& l = ps.layers[i-1];
//         int layer_id = i-1;
//         edge_policy.clear();
//         int num_edges = s.block[i]->indices.size();
// 	// std::cout << "Get edge policy\n";
//         this->get_edge_policy(s.block[i-1]->layer_nds,  *s.block[i], edge_policy, i-1, s.num_layers );
// 	// std::cout << "start slicing \n";
// 	this->slice_layer(s.block[i-1]->layer_nds, \
//           (* s.block[i]), l, layer_id, edge_policy);
// 	// std::cout << "Start reordering\n";
// 	this->reorder(l);
//         int p_edges = 0;
//         for(int j=0;j<this->num_gpus;j++){
//           p_edges += l.bipartite[j]->indices_L.size();
//           p_edges += l.bipartite[j]->indices_R.size();
//           if((l.bipartite[j]->indices_R.size()==0) || (l.bipartite[j]->indices_L.size() == 0)){
//             check = true;
// //std::cout << "Found empty remote graph" << l.bipartite[j]->indices_R.size() <<" " << l.bipartite[j]->indices_L.size() <<"\n";
// 	  }
// 	//if(!this->self_edge){
//         //    p_edges += l.bipartite[j]->self_ids_offset;
//         //  }
//         }
//         assert(num_edges == p_edges);
//         // l.debug();
//     }
//     for(int i=0;i<this->num_gpus;i++){
//         ps.cache_miss_from[i].clear();
//         ps.cache_hit_from[i].clear();
//         ps.cache_miss_to[i].clear();
//         ps.cache_hit_to[i].clear();
//         ps.last_layer_nodes[i].clear();
//           // ps.layers[s.num_layers-1].bipartite[i]->debug();
//         for(int j = 0; j <ps.layers[s.num_layers- 1].bipartite[i]->in_nodes.size(); j++){
//           auto nd = ps.layers[s.num_layers-1].bipartite[i]->in_nodes[j];
//           if (this->storage_map[i][nd] != -1){
//               ps.cache_hit_from[i].push_back(this->storage_map[i][nd]);
//               ps.cache_hit_to[i].push_back(j);
//           }else{
//             ps.cache_miss_from[i].push_back(nd);
//             ps.cache_miss_to[i].push_back(j);
//           }
//         }
//         vector<long> &last_layer = ps.layers[0].bipartite[i]->out_nodes_local;
//         ps.last_layer_nodes[i].insert(ps.last_layer_nodes[i].end(), last_layer.begin(), last_layer.end());
//     }
//     edge_policy.clear();
// 	if(check){
// // std::cout << "Skipping Cross checking if there are actually 0 remote subgraphs\n";
// 	//s.check_remote(this->workload_map);
// 	//test_sample_partition_consistency_gat(s,ps, \
// 	//   this->storage, this->gpu_capacity, this->workload_map.size());
//     }
//   }
//
//
//
//   void Slice::measure_pull_benefits(Sample &s){
//       int num_nodes = this->workload_map.size();
//       for(int i= 1; i< s.num_layers + 1;i++){
//          vector<int> p_indegree[this->num_gpus];
//          vector<int> p_outdegree[this->num_gpus];
//          vector<int> p_pulled[this->num_gpus];
//          vector<int> p_new_indegree[this->num_gpus];
//          vector<long>& in_nodes = s.block[i-1]->layer_nds;
//          vector<long>& offsets = s.block[i]->offsets;
//          vector<long>& indices = s.block[i]->indices;
//          for(int i=0;i<4;i++){
//            p_indegree[i].resize(num_nodes);
//            p_outdegree[i].resize(num_nodes);
//            p_pulled[i].resize(num_nodes);
//            p_new_indegree[i].resize(num_nodes);
//            memset(p_indegree[i].data(), 0, sizeof(int) * num_nodes);
//            memset(p_outdegree[i].data(), 0, sizeof(int) * num_nodes);
//            memset(p_pulled[i].data(), 0, sizeof(int) * num_nodes);
//            memset(p_new_indegree[i].data(),0, sizeof(int) * num_nodes);
//         }
//          for (int i = 0;i < in_nodes.size(); i++){
//            auto nd1 = in_nodes[i];
//            int p_dest = this->workload_map[nd1];
//            for(int j = offsets[i]; j < offsets[i+1]; j++){
//               auto nd2 = indices[j];
//               int p_src = this->workload_map[nd2];
//               if(p_src == p_dest) continue;
//               p_indegree[p_src][nd1] ++ ;
//               p_outdegree[p_dest][nd2]++;
//            }
//          }
//         int shuffle_cost  = 0;
//         for (int i=0;i<in_nodes.size();i++){
//           auto nd1 = in_nodes[i];
//           int p_dest = this->workload_map[nd1];
//           for(int j=0; j<4; j++){
//               if((p_dest != j) && (p_indegree[j][nd1] > 0))shuffle_cost ++;
//           }
//         }
//         std::cout << "Total shuffle cost" << shuffle_cost <<"\n";
//         for (int i = 0;i < in_nodes.size(); i++){
//           auto nd1 = in_nodes[i];
//           int p_dest = this->workload_map[nd1];
//           for(int j = offsets[i]; j < offsets[i+1]; j++){
//              auto nd2 = indices[j];
//              int p_src = this->workload_map[nd2];
//              if((p_src != p_dest) && (p_outdegree[p_dest][nd2] > p_indegree[p_src][nd1]/4)){
//                p_pulled[p_dest][nd2] = 1;
//              }
//           }
//         }
//         for (int i=0;i<in_nodes.size();i++){
//           auto nd1 = in_nodes[i];
//           int p_dest = this->workload_map[nd1];
//           for(int j = offsets[i]; j < offsets[i+1]; j++){
//              auto nd2 = indices[j];
//              int p_src = this->workload_map[nd2];
//              if((p_dest != p_src) && (p_pulled[p_dest][nd2] != 1)) p_new_indegree[p_src][nd1] ++ ;
//           }
//         }
//         int pull = 0;
//         for(int i=0;i<num_nodes;i++){
//           for(int j=0;j<4;j++){
//              if(p_pulled[j][i]==1)pull++;
//           }
//         }
//         shuffle_cost  = 0;
//         for (int i=0;i<in_nodes.size();i++){
//           auto nd1 = in_nodes[i];
//           int p_dest = this->workload_map[nd1];
//           for(int j=0; j<4; j++){
//               if((p_dest != j) && (p_new_indegree[j][nd1] > 0))shuffle_cost ++;
//           }
//         }
//         memset(p_indegree[0].data(), 0, sizeof(int) * num_nodes);
//         for(auto i: indices){
//           p_indegree[0][i] = 1;
//         }
//         int total = 0;
//         for(int i=0;i<num_nodes; i++){
//           total += p_indegree[0][i];
//         }
//         std::cout << "Total hybrid cost" << shuffle_cost <<":"<< pull <<":" <<total<<"\n";
//       }
//   }
