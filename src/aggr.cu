// Naive global aggregation
__global__ void aggregate_local(ComputeGraph *cgr,int *in, int *out){
    int threadIdx;
    int fIdx;
    int nbs = cgr->offsets[threadIdx]
    int s = 0;
    for(int i=0;i<nbs;i++){
        s = s + cgr->edges[i];
    }
    out[fIdx] = s;
}

// Initially compute graph is on the cpu.
void initializeSpaceOut(ComputeGraph *cgr){
    cudaMalloc()
}
