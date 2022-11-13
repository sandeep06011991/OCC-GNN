

void block_merge(int *b1, int * b2, int *out){
	int threadId = threadIdx.x;
	int blockId = blockIdx.x;
	int i = 16; 

	__shared__ int m[32];
	while(i > 0){
		if (threadId < i){
		
		
		}
	}

}


