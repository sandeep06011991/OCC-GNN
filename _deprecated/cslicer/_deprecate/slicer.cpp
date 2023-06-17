
#include "slicer.h"
#include <stdlib.h>
#include <iostream>


int  Slicer::neighbour_sample(long nd1, vector<long>& neighbors){
  neighbors.push_back(nd1);
  long nbs = this->dataset->indptr[nd1+1] - this->dataset->indptr[nd1];
  int offset = this->dataset->indptr[nd1];
  int in_degree = 0;
  if((nbs < 10) || (this->deterministic)){
    in_degree = nbs;
    for(int i=0;i<nbs;i++){
      neighbors.push_back(this->dataset->indices[offset + i]);
    }
  }else{
    in_degree = 10;
    for(int i=0;i<10;i++){
      int rand_nb = this->random_number_engine()%nbs;
      neighbors.push_back(this->dataset->indices[offset + rand_nb ]);
    }
  }
  if (in_degree == 0){
    // To avoid zero/zero division when divide
    in_degree = 1;
  }

  return in_degree;
}


void Slicer::slice_layer(vector<long>& in, vector<long>& out, Layer& l, int layer_id){
    for(long nd1: in){
      neighbors.clear();
      int in_degree = neighbour_sample(nd1, neighbors);
      long nbs = this->dataset->indptr[nd1+1] - this->dataset->indptr[nd1];
      int to = (*this->workload)[nd1];
      int t[4];
      for(int ii = 0;ii<4; ii ++ ){
        t[ii] = 0;
      }
      for(long nd2 : neighbors){
        if(nd1 == nd2){
            l.bipartite[to]->add_self_edge(nd1, in_degree);
        }else{
            int from = (*this->workload)[nd2];
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
        if(this->out_dr->mask[nd2]==0){
          out.push_back(nd2);
          this->out_dr->mask[nd2]=1;
          this->out_dr->used_nodes.push_back(nd2);
        }

      }
    }

    if(layer_id != 2){
      for(int i=0;i<4;i++){
        l.bipartite[i]->reorder(dr);
      }
    }else{
      for(int i=0;i<4;i++){
          l.bipartite[i]->reorder_lastlayer(dr,*storage_map[i], gpu_capacity[i]);
      }
    }
    this->out_dr->clear();
    if (this->deterministic){
      layer_consistency(l);
    }
  }

void Slicer::layer_consistency(Layer& l){
  for(int a=0; a<4; a++ ){
    BiPartite *send = l.bipartite[a];
    for(int b =0; b < 4; b ++){
    	if (a==b)continue;
	BiPartite *recv= l.bipartite[b];
	if(send->to_ids[b].size() != recv->from_ids[a].size()){
		std::cout << "FAILURE ALERT !!!!!!!!!!!!!!\n";
	}
	//cout << "send gpu:" << a << "shape" << send->to_ids[b].size() << "to_ids gpu:" << b << "shape" << recv->from_ids[a].size() <<"\n";
	assert(send->to_ids[b].size() == recv->from_ids[a].size());

    }
  }
 }
/*  Used this code for debugging structure of bipartite graphs
 *
    for(long o_id = 0; o_id<bp->owned_out_nodes.size() ;o_id++ ){
        // local aggregation
        long nd = bp->owned_out_nodes[o_id];
        if(bp->out_nodes[nd] == 168394)std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n";
        if(bp->out_nodes[nd] == 168394)std::cout << "working with" << bp->out_nodes[nd] << " ";
        long start = bp->indptr[nd];
        long end =  bp->indptr[nd  + 1];
        long local_nbs = end - start;
        if(bp->out_nodes[nd] == 168394)std::cout << "local nbs" << local_nbs <<" ";
        for(int gpu_r = 0;gpu_r<4 ; gpu_r++){
          if(gpu_r ==gpu)continue;
          for(int p_id= 0;p_id < bp->from_ids[gpu_r].size(); p_id ++ ){
            long remote_nd = bp->from_ids[gpu_r][p_id];
            if(remote_nd == nd){
              long remote_id = l.bipartite[gpu_r]->to_ids[gpu][p_id];
              long remote_nbs = (l.bipartite[gpu_r]->indptr[remote_id+1] - l.bipartite[gpu_r]->indptr[remote_id]);
              if(bp->out_nodes[nd] == 168394)std::cout << "found remote" << remote_nbs << " ";
              // long offset = l.bipartite[gpu_r]->indptr[remote_id];
              // for(long ii = 0;ii < remote_nbs; ii ++ ){
              //   std::cout <<l.bipartite[gpu_r]->in_nodes[l.bipartite[gpu_r]->indices[offset + ii]] << " ";
              // }
              local_nbs += remote_nbs;
            }
          }
        }
        if(bp->out_nodes[nd] == 168394)std::cout << "\n";
        if(local_nbs == 0){
          if(bp->in_degree[o_id] != 1){
            std::cout << "Degree" << bp->in_degree[o_id] <<" "<< local_nbs <<"\n";
          }
        }else{
          if(bp->in_degree[o_id]!=local_nbs){
            std::cout << "Degree" << bp->in_degree[o_id] <<" "<< local_nbs <<"\n";
          }
          // assert(bp->in_degree[o_id] == local_nbs);
        }
    }

}
*/

// Challenge moving parts.
// Dont worry about overlap. Just move this.
// Key challenge. I am mixing performance and variable expressibility.
  void Slicer::get_sample(vector<long> &batch){
    int batch_size = batch.size();
    in.clear();
    for(int i=0;i<batch_size;i++){
      in.push_back(batch[i]);
    }
    out.clear();
    for(int i=0;i<3;i++){
      // std::cout << "Sample layer" <<in.size() <<"\n";
      this->slice_layer(in, out, sample.layers[i],i);
      in.clear();
      in = out;
      out.clear();
    }

}

void Slicer::simple_3_hop_sample(int batch_id){
  assert(batch_id * batch_size < num_nodes);
  long * in_start = &this->target_nodes[batch_id * batch_size];
  long remaining = this->num_nodes - (batch_id * batch_size);
  if(remaining > this->batch_size){
    remaining = batch_size;
  }
  int batch_size = remaining;
  in.clear();
  for(int i=0;i<batch_size;i++){
    in.push_back(in_start[i]);
  }
  for(int i=0;i<3;i++){
    for(long nd1: in){
      this->neighbour_sample(nd1, nbs);
      for(int j=0;j<nbs.size();j++){
        sample.layers[i].bipartite[0]->add_edge(nd1,nbs[j],true);
      }
      out.insert(out.end(), nbs.begin(), nbs.end());
      nbs.clear();
    }
    sample.layers[i].bipartite[0]->reorder(dr);
    this->dr->order_and_remove_duplicates(out);
    this->dr->clear();
    in.clear();
    in = out;
    out.clear();
  }
}


void Slicer::clear(){
  for(int i=0;i<3;i++){
    this->sample.layers[i].clear();
  }
}
/*
void Slicer::run(){
  while(true){
    std::vector<long> * queue = this->work_queue->pop_object();
    // std::cout << "worker popeed \n";
    if(queue->size()==0){
      // Serves end of sample signal.
      std::cout << "recieved end sizgnal";
      delete queue;
      return;
    }
    clear();
    get_sample(*queue);
    /* std::cout << "sample created \n";
    for(int i=0;i<3;i++){
    	std::cout << "Sample layers\n";
	for(int j=0;j<4;j++){
		std::cout << "Bipartite graph\n";
		this->sample.layers[i].bipartite[j]->debug();
	}
    }
    // Sample *sample1 = new Sample();
    // PySample *sample = new PySample(sample1);
    PySample *sample = new PySample(&this->sample);
    this->generated_samples->push_object(sample);
    delete queue;
  }
}
*/
