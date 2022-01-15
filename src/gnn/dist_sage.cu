
#include "util/dist_tensor.hh"
#include "gnn/dist_sage.hh"
#include <vector>
#include "util/tensor.hh"
#include "util/gpu.hh"
#include "util/timer.h"

void DistSageAggr::forward(vector<int>& ind_ptr, vector<int>& indices,
          DistTensor& in, int num_nodes_out, int num_nodes_in, int *ext_map){
      // some magic reordering map
      // comes in externally.
      // std::cout << "v1\n";
      int * reorder_map = (int *)malloc(sizeof(int) * num_nodes_out);
      int count = 0;
      int min_mov = 0;
      int l = 0;
      if(!this->isExternalPartitioning){
        if(this->israndomPartitioning){
          for(int i=0; i< num_nodes_out;i++){
            reorder_map[i] = i%no_gpus;
          }
        }else{
          int max_gpu[4];
          for(int i=0; i< num_nodes_out;i++){
            int start = ind_ptr[i];
            int end = ind_ptr[i+1];
            memset(max_gpu,0,sizeof(int) * 4);
            for(int j = start;j<end;j++){
              max_gpu[in.global_to_gpu[indices[j]]]++;
            }
            int max_g = 0;
            for(int j=1;j<4;j++){
              if(max_gpu[j] > max_gpu[max_g])max_g = j;
            }
            reorder_map[i] = max_g;
            if(reorder_map[i]!=ext_map[i])count++;
            for(int j=0;j<4;j++){
              if(j == max_g) continue;
              min_mov = min_mov + max_gpu[j];
              if(max_gpu[j]>0)l++;
            }

          }
        }
    }else{
      for(int i=0; i< num_nodes_out;i++){
        reorder_map[i] = ext_map[i];
      }
    }
    // std::cout << count <<" " << min_mov << " "
    //     << num_nodes_in << " "<<  num_nodes_out <<" " << l << "\n";
      if(this->out_feat != nullptr){
        this->out_feat->clearTensor();
        delete (this->out_feat);
      }
      // std::cout << "v2\n";
      struct Shape s;
      s.dim1 = num_nodes_out;
      s.dim2 = in.s.dim2;

      this->out_feat = new DistTensor(s , reorder_map, no_gpus);

      // std::cout << "v3\n";
      // Populate remote_csrs.
      this->populateLocalGraphs(in, ind_ptr, indices);
      // std::cout << "v4\n";
      for(int i=0;i<no_gpus;i++){

        for(int j=0;j<no_gpus;j++){
          if(i==j){
            this->local_graph[i][j].create_local_csr(this->out_feat->local_to_global[i].size());
          }else{
            this->local_graph[i][j].create_remote_csr();
          }
        }
      }
      sync_all_gpus();
      // std::cout << "v5\n";
      // start_timer(MOVEMENT_COMPUTE1);
     // auto s1 = high_resolution_clock::now();
      for(int i=0;i<no_gpus;i++){
        for(int j=0;j<no_gpus;j++){
          this->local_graph[i][j].forward(*(in.local_tensors[i]));
        }
      }
      // std::cout << "v6\n";
	    //auto e1 = high_resolution_clock::now();
      //auto duration = (float) (std::chrono::duration_cast<std::chrono::milliseconds>(s1-e1).count())/1000;
      //std::count << "time " << duration <<"\n";
      // stop_timer(MOVEMENT_COMPUTE1);
      // sync_all_gpus();
      // Create temporary tensors and clean up after wards.
      Tensor<float> * temp[4][4];
      start_timer(MOVEMENT_COST);
      for(int src=0;src<no_gpus;src++){
        for(int dest=0;dest<no_gpus;dest++){
          if(src!=dest) {
            temp[src][dest] =  new Tensor<float>
                          (this->local_graph[src][dest].out,
                            this->local_graph[src][dest].dest_gpu);
          }else{
            temp[src][dest] = this->local_graph[src][dest].out;
          }
        }
      }
      stop_timer(MOVEMENT_COST);
      // std::cout << "v7\n";
      // sync_all_gpus();
      // start_timer(MOVEMENT_COMPUTE1);
      for(int dest=0;dest<no_gpus;dest++){
        for(int src=0;src<no_gpus;src++){
          if(src!=dest) {
            merge(temp[src][dest],temp[dest][dest],
              this->local_graph[src][dest].local_to_local_t);

          }
        }
      }
      // sync_all_gpus();
      // stop_timer(MOVEMENT_COMPUTE1);
      // std::cout << "v8\n";
      // out = new DistributedTensor(reorderer_map,shape);
      for(int i=0;i<no_gpus;i++){
        this->out_feat->local_tensors[i] = temp[i][i];
      }
      for(int src=0;src<no_gpus;src++){
        for(int dest=0;dest<no_gpus;dest++){
          if(src!=dest) {
            temp[src][dest]->clearTensor();
            delete(temp[src][dest]);
          }
        }
      }
      // std::cout << "v9\n";
}


void DistSageAggr::populateLocalGraphs(DistTensor &in, vector<int> &indptr,
                                          vector<int> &indices){
  for(int i=0;i<no_gpus;i++){
    for(int j=0;j<no_gpus;j++){
      this->local_graph[i][j].clear();
    }
  }
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

__global__ void mergeKernel(float *src, int src_dim1, int src_dim2,
        float *dest, int dest_dim1, int dest_dim2,  int * indices, int size){
    int x = blockIdx.x;
    int y = threadIdx.x;
    assert(indices[x] < dest_dim1);
    dest[indices[x] * src_dim2 + y] += src[x * src_dim2 + y];
}


void merge(Tensor<float> *src, Tensor<float> *dest, Tensor<int> *indices){
  assert(src->device_id == dest->device_id);
  assert(indices->device_id == dest->device_id);
  assert(src->device_id == dest->device_id);
  assert(indices->s.dim1 == src->s.dim1);
  int noThreads = src->s.dim2;
  int noBlocks = src->s.dim1;
  cudaSetDevice(src->device_id);
  cudaEvent_t start;
  cudaEvent_t stop;
  auto error = cudaEventCreate(&start);
  cudaEventCreate(&stop);
  auto  error_1 = cudaEventRecord(start);
  mergeKernel<<<noBlocks,noThreads>>>(src->data_device, src->s.dim1, src->s.dim2,
                          dest->data_device, dest->s.dim1, dest->s.dim2,
                          indices->data_device, indices->s.dim1);
   auto error1 = cudaEventRecord(stop);
   auto error2 = cudaEventSynchronize(stop);
   float msec = 0.0f;
   auto error3 = cudaEventElapsedTime(&msec, start, stop);
   add_timer_ms(MOVEMENT_COMPUTE2,msec);
   cudaEventDestroy(start);
   cudaEventDestroy(stop);
  // cudaDeviceSynchronize();
  NNException::throwIfDeviceErrorsOccurred("Failed Merge kernel \n");
}
