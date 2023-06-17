Each py-worker contains one cpp-slicer.
  Why didnt I have one py-worker and multiple cpp-slicer.
  This is because heterograph graph construction could be potentially a bottleneck.

Layout:
1. Dataset.cpp/h = reads the graph into binaries. Better to read from binaries than to pass large data structures.
2. slicer.h/cpp = Inputs(storage_map and workload_map)
  Takes in dataset, storage map
  === neighbour_sample
    Samples neighbourhood of one vector.
  === slice_layer
    for each nd samples and then slices into bipartite graph.
  ===
  Sample ==> Slice[Take into account] ==> Pythonize
3. pyfrontend.cpp
4. pybipartite.h
5. duplicate.cpp/h

Problem: Logic for  slicing and serialization is coupled too tightly. 



ToDo:
1. Move WorkerPool.cpp and WorkerPool.h

Goal of this refactorization.
1. Move all deprecated code, add modularization and perf tests.  
2. Make neighbour fan out a variable.
