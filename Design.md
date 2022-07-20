#### Understanding the codebase.

1. Understand the input files.
     python frontend utils/utils.py
     cpp frontend src/dataset.cpp, src/dataset.h

2. Understand the memory manager. 
     python/utils/memory_manager.py

3. Link Sampler and Memory Manager.  

#### Design and sprint plan

Total time 40 hour week. Potential targets Pagraph, P3, DGCL.
Other variants possible are GNN-Autoscale and LazyGCN
Focus on beating Pagraph through better cache utilization for deeper and residual gcn.    
Biggest weakness, is with larger graphs where 2hop neighbour hood doesnt work.   

1. Get datasets: Download dataset from pagraph convert to binary files for quick loading (4 hours)
2. GCN and graphsage single node training (1 day)
3. GCN with neighbour sampling single node training(1 day)
4. GCN with sampling multigpu with data shuffling training (1 day)
  Most basic approach to sampled GCN training.

5. GCN with partition data. (1 day)
6. Sampled training on partitioned data with computation flow (1 day)
