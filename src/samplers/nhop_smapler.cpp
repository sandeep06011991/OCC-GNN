// #include "samplers/nhop_sampler.h"

// void NhopSampler::get_sample(int batchId, int khop=2){
//     sample.clear();
//     sample.init(khop);
//     assert(batchId * this->minibatch_size < this->no_nodes);
//     int * tgt = &this->target_nodes[this->minibatch_size * batchId];
//     int no_nodes =  minibatch_size;
//     if(this->minibatch_size * (batchId + 1) > this->no_nodes){
//       no_nodes = this->no_nodes - (this->minibatch_size * batchId);
//     }
//     this->current_minibatch_size = no_nodes;
//     int nlayer = 0;
//     while(nlayer < khop){
//         for (int v_idx = 0; v_idx < no_nodes; v_idx++)
//         {
//             /* code */
//             int v = tgt[v_idx];
//             int v_begin = graph.indptr[v];
//             int v_end = graph.indptr[v+1];
//             int v_deg = v_end - v_begin;

//             sample.add_v(v, nlayer);

//             int max_sample_size = get_sample_size(nlayer);
//             if (v_deg < max_sample_size){
//                 for (int n_idx = v_begin; n_idx < v_end; n_idx++)
//                 {
//                     int n = graph.indices[n_idx];
//                     sample.add_n(n, nlayer);
//                     sample.add_edge(v, n, nlayer);
//                 }
//             } else {
//                 for (int i = 0; i < max_sample_size; i++)
//                 {
//                     int rand_idx = rand() % v_deg;
//                     int n = graph.indices[v_begin + rand_idx];
//                     sample.add_n(n, nlayer);
//                     sample.add_edge(v, n, nlayer);
//                 }
//             };
//         }
//         // move to next layer
//         sample.remove_duplicates(nlayer);
//         sample.create_csr(nlayer);
//         nlayer += 1;
//     }
//     fill_batch_data();
// };
