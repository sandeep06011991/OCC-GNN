

void aggregate(std::vector<int>& layer_ns,
        std::vector<int> &offsets, std::vector<int> &indices, std::vector<int> &degree \
        std::vector<int> &in, std::vector<int> &out, bool first_layer){
  for(int i=0;i < offsets.size()-1; i++){
    int start = offsets[i];
    int end = offsets[j];
    int src = layer_nds[i];
    int t = 0;
    for(int j = start; j < end; j++){
      int dest = indices[j];
      acc = dest;
      if(!first_layer){
        acc = in[dest];
      }
      if (src == dest){
        acc = acc * in_degree[i]
      }
      t  += acc;
    }
    out[src] = acc;
}


// Sample without any reordering.
// returns sum of flow up after all layers.
int sample_flow_up(Sample &s, int number_of_nodes){
   std::vector<int> in_f(number_of_nodes);
   std::vector<int> out_f(number_of_nodes);
   for(int i=s.num_layers; i >=0; i--){
     bool first_layer = (i == s.num_layers);
     aggregate(s.blocks[i]->layer_nds, s.blocks[i+1]->offsets, s.blocks[i+1]->indices, \
          s.blocks[i+1]->in_degrees, in_f, out_f, first_layer);
     in_f.swap(out_f);
   }
   s = 0;
   for(int i: s.blocks[0]){
     s += out_f[in[0]];
   }
   return s;
}

void aggregate(vector<int> &out, vector<int> &in, Bipartite *bp){
    vector<int> & indptr = bp->indptr;
    vector<int> &indices = bp->indices;
    for(int i=0;i<indptr.size();i ++){
      int off_start = indptr[i];
      int off_end = indptr[i+1];
      int t = 0;
      for(int off = off_start; off < off_end; off ++ ){
          t += in[indices[off]];
      }
      out[i] = t;
    }
}

void shuffle(vector<int>& from_ids,  vector<int> &from_v,
         vector<int> &to_ids,  vector<int> &to_v){
  assert(from_ids.size() == to_ids.size());
  for(int i=0; i< from_ids.size(); i++){
    from_v[from_ids[i]] += to_v[to_ids[i]];
  }
}

void pull_own_node(Bipartite *bp, vector<int> &out, vector<int> &in){

  assert(bp->self_ids_in.size() == bp->self_ids_out.size())
  for(int i=0; i < bp->self_ids_in.size(); i++){
    out[bp->self_ids_out[i]] += bp->in_degree[i] * in[bp->self_ids_in[i]];
  }
  in.resize(bp->owned_out_nodes.size());
  for(int i=0;i< bp->owned_out_nodes.size(); i++){
    in[i] = out[bp->owned_out_nodes[i]];
  }
}


// Partitioned flow must have same output.
void sample_flow_up(PartitionedSample &s,std::vector<int> test_storage_map[4]){
  // refresh storage map with local_ids.
  std::vector<int> in[4];
  std::vector<int> out[4];
  for(int i=0;i<4; i++ ){
    in[i].swap(test_storage_map[i]);
  }
  for(int i =  s.num_layers-1  ; i>=0 0; i--){
    // Bipartite local aggregation.
    PartitionedLayer &layer = s.layers[i];
    for(int j=0; j<4; j++ ){
        out[j].resize(layer.bipartite[j]->num_out_nodes);
        aggregate(out[j], in[j], layer.bipartite[j]);
    }
    // Shuffle aggregate
    for(int from = 0; from < 4; from ++) {
      for(int to = 0; to < 4 ; to++) {
          if(from != to){
            shuffle(layer.bipartite[from], out[from], layers[to], out[to]);
          }
      }
    }
    // Pull locally and add degree
    // Slice owned node.
    for(int j = 0; j < 4; j++){
      pull_own_node(layer.bipartite[j], out[j], in[j]);
    }
  }
  s = 0;
  for(int i=0;i < 4;i++){
    for(int k:in[i]){
      s += k;
    }
  }
  return s;
}



int main(){
  // Three random samples with a batch size of 4096.
  // three cache scenarios to test.
  // [0,25,100] = Cache scenario.
}
