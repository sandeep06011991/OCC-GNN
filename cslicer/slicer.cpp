#include "slicer.h"
#include <stdlib.h>
#include <iostream>


inline void Slicer::neighbour_sample(long nd1, vector<long>& neighbors){
  neighbors.push_back(nd1);
  long nbs = this->dataset->indptr[nd1+1] - this->dataset->indptr[nd1];
  int offset = this->dataset->indptr[nd1];
  if(nbs < 10){
    for(int i=0;i<nbs;i++){
      neighbors.push_back(this->dataset->indices[offset + i]);
    }
  }else{
    for(int i=0;i<10;i++){
      int rand_nb = this->random_number_engine()%nbs;
      // int rand_nb = this->rng_coin(gen) % nbs;
      // int rand_nb = i;
      neighbors.push_back(this->dataset->indices[offset + rand_nb ]);
    }
  }
}


void Slicer::slice_layer(vector<long>& in, vector<long>& out, Layer& l){
    for(long nd1: in){
      neighbors.clear();
      neighbour_sample(nd1, neighbors);
      // neighbors.clear();
      // neighbour_sample(nd1, neighbors);
      int to = this->workload[nd1];
      for(long nd2 : neighbors){
        if(nd1 == nd2){
              l.bipartite[to]->add_self_edge(nd1);
        }else{
            int from = this->workload[nd2];
            if(to == from){
              l.bipartite[to]->add_edge(nd1,nd2,true);
            }else{
              l.bipartite[from]->add_edge(nd1,nd2,false);
              l.bipartite[to]->add_to_node(nd1,to);
              l.bipartite[from]->add_from_node(nd1,from);
            }
        }
        if(this->out_dr->mask[nd2]==0){
          out.push_back(nd2);
          this->out_dr->mask[nd2]=1;
          this->out_dr->used_nodes.push_back(nd2);
        }
      }
    }
    for(int i=0;i<4;i++){
      l.bipartite[i]->reorder(dr);
    }
    this->out_dr->clear();
    // std::cout << "pre dup size" << out.size() <<"\n";
    // vector<long> backup = out;
    // this->dr->order_and_remove_duplicates(out);
    // this->dr->clear();
    // this->dr->order_and_remove_duplicates(backup);
    // // std::cout << "post dup size" << out.size() <<"\n";
    // this->dr->clear();
    // remove duplicates from nd2.
  }

// Challenge moving parts.
// Dont worry about overlap. Just move this.
// Key challenge. I am mixing performance and variable expressibility.
  void Slicer::get_sample(vector<long> &batch){
    int batch_size = batch_id.size();
    in.clear();
    for(int i=0;i<batch_size;i++){
      in.push_back(batch[i]);
    }
    for(int i=0;i<3;i++){
      this->slice_layer(in, out, sample.layers[i]);
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

void Slicer::run(){
  while(true){
    std::vector<long> * queue = this->work_queue.pop();
    if(queue.size()==0){
      // Serves end of sample signal.
      return;
    }
    clear();
    get_sample(*queue);
    this->generate_samples(new PySample(this->sample));
    delete(queue);
  }
}
