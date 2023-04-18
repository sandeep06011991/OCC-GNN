#pragma once
#include "../util/device_vector.h"
#include "../graph/sample.h"
#include "../graph/sliced_sample.h"

namespace cuslicer{

  struct Vector_From_Index{
      long * data;
      long offset;

      __device__
      inline void  add_position_offset(long val, long pos){
          assert(pos - 1 -offset >= 0);
          // -1 because positions are caclulated from inclusie sum
          data[pos - 1 - offset ] = val;
      }

      __device__
      inline void  add_value_offset(long val, long pos){
        assert(pos >= 0);
        data[pos] = val - offset;
      }
  };

class Slice{
protected:
  device_vector<int> sample_workload_map;
  device_vector<int> workload_map;
  device_vector<int> storage_map[8];
  device_vector<int> storage[8];
  device_vector<int> sample_partition;
  // Used for new node ordering
  void** storage_map_flattened;
  int gpu_capacity[8];
  int num_gpus = 4;
  DuplicateRemover *dr;
  // Use this for GAT
  bool pull_optimization = false;
  device_vector<int> cache_hit_mask;
  device_vector<int> cache_miss_mask;
  long num_nodes  = 0;
  cudaEvent_t event1;
  cudaEvent_t event2;
  cudaEvent_t event3;
  cudaEvent_t event4;
  cudaEvent_t event5;
  cudaEvent_t event6;
  cudaEvent_t event7;




public:
// Are all these options really needed.
  Slice(device_vector<int> workload_map,
      std::vector<int> storage[8],
        bool pull_optimization, int num_gpus){
    gpuErrchk(cudaEventCreate(&event1));
    gpuErrchk(cudaEventCreate(&event2));
    gpuErrchk(cudaEventCreate(&event3));
    gpuErrchk(cudaEventCreate(&event4));
    gpuErrchk(cudaEventCreate(&event5));
    gpuErrchk(cudaEventCreate(&event6));
    gpuErrchk(cudaEventCreate(&event7));


    this->workload_map = workload_map;
    this->num_gpus= num_gpus;
    long num_nodes = this->workload_map.size();
    this->num_nodes = num_nodes;
    assert(num_gpus <= 8);
    std::vector<int> _t1;
    std::vector<int> _t2;
    for(int i=0;i<num_gpus;i++){
      _t1.clear();
      _t2.clear();
      for(int j = 0; j < num_nodes; j++){
          _t1.push_back(-1);
      }
      _t2 = storage[i];
      int count  = 0;
      for(auto j:_t2){
          _t1[j] = count;
           count ++ ;
      }
      // Must be an lvalue
      auto _s1 = device_vector<int>(_t1);
      this->storage_map[i] = _s1;
      auto s2 = device_vector<int>(_t2);
      this->storage[i] = s2;
      gpu_capacity[i] = count;
    }
    void *t[8];
    gpuErrchk(cudaMalloc(&storage_map_flattened, num_gpus * sizeof(int *)));
    for(int i= 0; i < num_gpus; i++){
    	t[i] = this->storage_map[i].ptr();
      // std::cout << "Storage map size" << this->storage_map[i].size() <<"\n";
    }
    std::cout << "All storage maps moved\n";

    gpuErrchk(cudaMemcpy(storage_map_flattened, t, sizeof(int *) * num_gpus,\
      cudaMemcpyHostToDevice));

    dr = new ArrayMap(this->workload_map.size());
    this->pull_optimization = pull_optimization;
  }

  void slice_sample(Sample &s, PartitionedSample &ps);

  virtual void slice_layer(device_vector<long>& in, Block &bl, \
      PartitionedLayer& l, bool last_layer) = 0;


  // void get_edge_policy(vector<long> &in, Block &bl, vector<POLICY> &policy, int layer_id, int num_layers);

  void  reorder(PartitionedLayer &l) ;

  void fill_cache_hits_and_misses(PartitionedSample &ps, int gpu, device_vector<long> &in_nodes);

  virtual void resize_bipartite_graphs(PartitionedLayer &ps,int num_in_nodes, int num_out_nodes,\
    int num_edges) = 0;

};

class PushSlicer: public Slice{

public:
    // Contains information from the offsets array and exclusive sum.
    // Use to construct graphs from partitioned edges.
    // Must have one to one mapping from every object in bipartite graph
    struct LocalGraphInfo{
        Vector_From_Index in_nodes;
        Vector_From_Index out_nodes_local;
        Vector_From_Index out_nodes_remote;
        Vector_From_Index out_degree_local;
        Vector_From_Index indptr_L;
        Vector_From_Index indptr_R;
        long num_out_local;
        Vector_From_Index indices_L;
        Vector_From_Index indices_R;
        Vector_From_Index push_from_ids[MAX_GPUS];

    };

    LocalGraphInfo host_graph_info[MAX_GPUS];
    LocalGraphInfo * device_graph_info;

    PushSlicer(device_vector<int> workload_map,
        std::vector<int> storage[8],
          bool pull_optimization, int num_gpus):Slice(workload_map,
            storage, pull_optimization, num_gpus){
              gpuErrchk(cudaMalloc(&device_graph_info, sizeof(LocalGraphInfo) * this->num_gpus ));
    }

    void copy_graph_info(){
      gpuErrchk(cudaMemcpy(device_graph_info, host_graph_info,  sizeof(LocalGraphInfo) * this->num_gpus, cudaMemcpyHostToDevice));
    }

    void slice_layer(device_vector<long>& in, Block &bl, \
        PartitionedLayer& l, bool last_layer) ;

    void slice_layer_per_gpu(device_vector<long>& in, Block &bl, \
        PartitionedLayer& l, bool last_layer, int gpu);

    void resize_bipartite_graphs(PartitionedLayer &ps,int num_in_nodes, int num_out_nodes,
        int num_edges);
};

class PullSlicer: public Slice{

    struct LocalGraphInfo{
        Vector_From_Index in_nodes_local;
        Vector_From_Index out_nodes_local;
        Vector_From_Index out_degree_local;
        Vector_From_Index indptr_L;
        Vector_From_Index indices_L;
        long num_out_local;
        long num_in_local;
        long num_in_pulled;
        Vector_From_Index pull_to_ids[MAX_GPUS - 1];
    };

    LocalGraphInfo host_graph_info[MAX_GPUS];
    LocalGraphInfo * device_graph_info;

public:
    PullSlicer(device_vector<int> workload_map,
        std::vector<int>  storage[8],
          bool pull_optimization, int num_gpus):Slice(workload_map,
            storage, pull_optimization, num_gpus){
              gpuErrchk(cudaMalloc(&device_graph_info, sizeof(LocalGraphInfo) * this->num_gpus ));
            }

    void slice_layer(device_vector<long>& in, Block &bl, \
        PartitionedLayer& l, bool last_layer);

    void resize_bipartite_graphs(PartitionedLayer &ps,
      int num_in_nodes, int num_out_nodes,
          int num_edges);    

    void copy_graph_info(){
      gpuErrchk(cudaMemcpy(device_graph_info, host_graph_info,  sizeof(LocalGraphInfo) * this->num_gpus, cudaMemcpyHostToDevice));
    }

};


}
