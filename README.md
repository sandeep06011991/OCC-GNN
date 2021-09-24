### Traditional Sampled training.
Multi GPU GCN, using collection and train independently.

## Coding Blocks. (Each block doesnt take beyond 1 hour.)
1. Read datasets from dgl datasets (DONE!)
2. Populate data into graph cpp datastructures. (DONE)
3. Initialize data on GPU (DONE)
4. Add exception support (DONE)
4. Initialize model weights on gpu linear layers.
  Build Linear Layers.
    
5. Forward Pass.
6. Compute Loss
7. Backward Pass
8. Combine everything into training iteration.
9. Add the concept of a sample.
10. Create a sampled gcn (TOO COARSE)

## Credits:

Parts of the code have been inspired/copied from the following repos.
1. https://github.com/pwlnk/cuda-neural-network/
