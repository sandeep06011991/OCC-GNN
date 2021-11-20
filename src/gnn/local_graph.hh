// Contains information of edges present on src gpu id which are then sent to
// destination gpu id.
#include<vector>
#include "util/tensor.hh"

class LocalComputeGraph{

  std::vector<int> indptr;
  // temporary structure for ease of coding.
  std::vector<int> dest_local_id;
  std::vector<int> indices;
  // size of indptr - 1
  std::vector<int> local_to_local;
  int src_gpu;
  int dest_gpu;

  Tensor<float> *out = nullptr;
  SageAggr  * aggr;

public:

  void set_src_dest(int src_id,int dest_id){
    this->src_gpu = src_id;
    this->dest_gpu = dest_id;
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
  }

  void forward(Tensor<float> &src){
      Tensor<float>& forward(Tensor<int>& ind_ptr, Tensor<int>& indices,
              Tensor<float>& in, int num_nodes_out, int num_nodes_in);
      Tensor<int> * ind_ptr_t =
          new Tensor<int>(ind_ptr.data(),Shape(ind_ptr.size(),1),src_gpu_id);
      Tensor<int> * indices_t =
            new Tensor<int>(indices.data(), Shape(indices.size(),1), src_gpu_id);
      int num_nodes_out = ind_ptr.size()-1;
      int num_nodes_in = src.s.dim1;
      cudaSetDevice(src);
      out = aggr.forward(ind_ptr_t, indices_t, src, num_nodes_out, num_nodes_in);

  }

};
