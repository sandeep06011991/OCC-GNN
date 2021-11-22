// Contains information of edges present on src gpu id which are then sent to
// destination gpu id.
#include<vector>
#include "util/tensor.hh"
#include "gnn/sage.hh"
// Global data graph is broken down into local executable units on individual gpus.
// Main inputs are dest_local_ids and indices and local_to_local.
// populated using add_edge.
// ind ptr only for convenience
class LocalComputeGraph{

  std::vector<int> indptr;
  // temporary structure for ease of coding.
  std::vector<int> dest_local_id;
  std::vector<int> indices;
  // size of indptr - 1
  std::vector<int> local_to_local;
  int src_gpu;
  int dest_gpu;

  Tensor<float> out;
  SageAggr  * aggr;
  Tensor<int> * ind_ptr_t;
  Tensor<int> * indices_t;
  Tensor<int> * local_to_local_t;

public:

  void set_src_dest(int src_id,int dest_id,int fsize){
    this->src_gpu = src_id;
    this->dest_gpu = dest_id;
    aggr = new SageAggr(fsize);
  };

  void create_csr(){
    int prev = dest_local_id.at(0);
    int start = 0;
    indptr.push_back(start);
    int end = start+1;
    for(int i=1;i < dest_local_id.size();i++){
      if(dest_local_id[i]!=prev){
        indptr.push_back(end);
        local_to_local.push_back(prev);
        prev = dest_local_id[i];
        start = end;
        end = start;
      }
      end = end + 1;
    }
    local_to_local.push_back(prev);
    indptr.push_back(end);
    assert(indptr.size()== (local_to_local.size()+1));

  }

  void add_edge( int src_local_id,int dest_local_id){
    indices.push_back(src_local_id);
    this->dest_local_id.push_back(dest_local_id);
  }

  void clear(){
    indptr.clear();
    dest_local_id.clear();
    indices.clear();
    local_to_local.clear();
    if(ind_ptr_t!=nullptr){
      ind_ptr_t->clearTensor();
      delete ind_ptr_t;
    }
    if(indices_t!=nullptr){
      indices_t->clearTensor();
      delete ind_ptr_t;
    }
    if(local_to_local_t!=nullptr){
      local_to_local_t->clearTensor();
      delete local_to_local_t;
    }
  }

  void forward(Tensor<float> &src){
      // Tensor<float>& forward(Tensor<int>& ind_ptr, Tensor<int>& indices,
      //         Tensor<float>& in, int num_nodes_out, int num_nodes_in);
      ind_ptr_t = new Tensor<int>(indptr.data(),
                      Shape(indptr.size(),1),src_gpu);
      indices_t = new Tensor<int>(indices.data(),
                      Shape(indices.size(),1), src_gpu);
      local_to_local_t = new Tensor<int>(local_to_local.data(),
                      Shape(local_to_local.size(),1),dest_gpu);
      int num_nodes_out = indptr.size()-1;
      int num_nodes_in = src.s.dim1;
      cudaSetDevice(src_gpu);
      out = aggr->forward(*ind_ptr_t, *indices_t, src, num_nodes_out, num_nodes_in);

  }

};
