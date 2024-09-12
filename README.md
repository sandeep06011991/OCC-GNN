# README 

## Optimized computation through communication (OCC) 
### (later named GSplit)

This is repository is the early prototype for the paper GSplit: Scaling Graph Neural Network Training on Large Graphs via Split-Parallelism.
Split introduces hybrid parallel mini-batch training paradigm called split parallelism. 
Split parallelism splits the sampling and training of each mini-batch across multiple GPUs online, at each iteration, using a lightweight splitting algorithm.
This was proposed as an nvidia research proposal in Dec 2021 work was awarded the [NSF grant](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2224054&HistoricalAwards=false) on July 22.
The current branch is the working prototype for the [arxiv](https://arxiv.org/abs/2303.13775)](V1) 
which introduces cooperative training while sampling on the CPU and splitting the training phase across GPUs with its last working commit dated to May 2022.
This version was submitted to  working prototype has its last commit dated to May 2022 and submitted to OSDI Dec 2022(rejected).
This work was further optimized in our current V2 submission to perform split sampling along with probabilistic guarantees.


Concurrent work done by M.F Balin done during his nvidia internship was discovered by us as also introducing split training with further split parallel sampling shown on the [pull request]
(Muhammed Fatih Balin, Dominique LaSalle, and Ümit V. Çatalyürek. Cooperative minibatching, August 2022. URL https://github.com/dmlc/dgl/pull/4337) dated Aug 2022. submitted to ICML in Jan 2023 and which was subsequently submitted to [arxiv](https://arxiv.org/abs/2310.12403).