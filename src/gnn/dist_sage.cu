#include "util/dist_tensor.hh"
#include "gnn/dist_sage.hh"
#include <vector>
#include "util/tensor.hh"

void DistSageAggr::forward(vector<int>& ind_ptr, vector<int>& indices,
          DistTensor& in, int num_nodes_out, int num_nodes_in){
      // some magic reordering map
      int * reorder_map = (int *)malloc(sizeof(int) * num_nodes_out);
      for(int i=0; i< num_nodes_out;i++){
        reorder_map[i] = i%2;
      }

      if(this->out_feat == nullptr){
        this->out_feat->clearTensor();
        free(out_feat);
      }
      struct Shape s;
      s.dim1 = num_nodes_out;
      s.dim2 = in.s.dim2;
      out_feat = new DistTensor(s , reorder_map);

      // Populate remote_csrs.
      populateLocalGraphs(in, ind_ptr, indices);
      for(int i=0;i<no_gpus;i++){
        for(int j=0;j<no_gpus;j++){
        this->local_graph[i][j].create_csr();
        }
      }

      for(int i=0;i<no_gpus;i++){
        for(int j=0;j<no_gpus;j++){
          this->local_graph[i][j].forward(*(in.local_tensors[i]));
        }
      }

      // Create temporary tensors and clean up after wards.
      Tensor<float> * temp[4][4];
      for(int src=0;src<no_gpus;src++){
        for(int dest=0;dest<no_gpus;dest++){
          if(src!=dest) {
            temp[src][dest] = new Tensor<float>
                          (this->local_graph[src][dest]->out,
                            this->local_graph[src][dest]->dest_gpu);
          }else{
            temp[src][dest] = this->local_graph[src][dest]->out;
          }
        }
      }
      sync_all_gpu();
      for(int dest=0;dest<no_gpus;dest++){
        for(int src=0;src<no_gpus;src++){
          if(src!=dest) {
            merge(temp[dest][dest],temp[src][dest],this->local_graph[src][dest].local_to_local[id]);
          }
        }
      }


      out = new DistributedTensor(reorderer_map,shape);
      for(int i=0;i<4;i++){
        out.local_tensors[i] = temp[i][i];
      }
      for(int src=0;src<no_gpus;src++){
        for(int dest=0;dest<no_gpus;dest++){
          if(src!=dest) {
            temp[src][dest]->cleanUpTensor();
            free(temp[src][dest]);
          }
      }

}


void DistSageAggr::populateLocalGraphs(DistTensor &in, vector<int> &indptr,
                                          vector<int> &indices){
  for(int i=0;i<indptr.size()-1;i++){
    int nd1 = i;
    int dest_gpu = this->out_feat->global_to_gpu[nd1];
    int dest_local_id = this->out_feat->global_to_local[nd1];
    assert(dest_gpu < this->no_gpus);
    int start = indptr[i];
    int end = indptr[i+1];
    for(int j=start;j<end;j++){
      int nd2 = indices.at(j);
      int src_gpu = in.global_to_gpu[nd2];
      int src_local_id = in.global_to_local[nd2];
      this->local_graph[src_gpu][dest_gpu].add_edge(src_local_id,dest_local_id);
    }
  }
}

__global__ mergeKernel(float *src, int src_dim1, int src_dim2,
        float *dest, int dest_dim1, int dest_dim2,  int * indices, int size){
    int x = blockIdx.x;
    int y = threadIdx.x;
    dest[indices[x] * src_dim2 + y] += src[x * src_dim2 + y];
}


void merge(Tensor<float> *src, Tensor<float> *dest, Tensor<int> indices){
  assert(src->gpu_id == dest->gpu_id);
  assert(indices->gpu_id == dest->gpu_id);
  assert(src->gpu_id == dest->gpu_id);
  int noThreads = src.s.dim2;
  int noBlocks = src.s.dim1;
  cudaSetDevice(src->gpu_id);
  <<<noBlocks,noThreads>>> mergeKernel(src->data, src.s.dim1, src.s.dim2,
                          dest->data, dest.s.dim1, dest.s.dim2,
                          indinces->data, indices.s.dim1);
}
