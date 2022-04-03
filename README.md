### Artifact Instructions.

We rely on the following datasets.

Dataset       | Source
------------- | -------------
OGBN-products  | DGL Package
OGBN-100mag | DGL Package
Reddit | DGL Package
com-orkut | SNAP
com-reddit | SNAP
com-friendster | SNAP

The steps to perform data preprocessing are approximately:
1. For DGL datasets.
    run utils/convert_dgl_dataset.py with appropriate ROOT_DIR and TARGET_DIR
    This step downloads the dgl datasets and create binaries for consumption in later
    phases.
2. For SNAP datasets.
    Download ungraph.txt.zip files from SNAP database.
    gunzip -d the text file with the correct directory name.
    run utils/convert_snap_dataset.py to create binaries for later consumption.
3. 
Multi GPU GCN, using collection and train independently.

## Coding Blocks. (Each block doesnt take beyond 1 hour.)
1. Read datasets from dgl datasets (DONE!)
2. Populate data into graph cpp datastructures. (DONE)
3. Initialize data on GPU (DONE)
4. Add exception support (DONE)
5. Initialize model weights on gpu linear layers.
  Build Linear Layers.

5. Forward Pass.
6. Compute Loss
7. Backward Pass
8. Combine everything into training iteration.
9. Add the concept of a sample.
10. Create a sampled gcn (TOO COARSE)
11. Compare with dgl with neighbourhood sampling.



## Credits:

## Important Datastructures and and design philosphy

1. Tensor:
    Located in either cpu or one of the gpus as a continuos array.
    Can be initialized from another tensor or cpu_array.
    Tensor creates its own pointers and frees them itself.
    It does not hold any external pointers.
    Can be instantiated as a blank or with values from another tensor.

2. Distributed Tensor
    A Global tensor which is split across multiple gpus.



Parts of the code have been inspired/copied from the following repos.
1. https://github.com/pwlnk/cuda-neural-network/
2. https://github.com/GT-TDAlab/MG-GCN
3. Understad the code base of quiver, pagraph and DGCL.
