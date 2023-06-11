import torch 
from .measure import *
import time 
import logging 
from quiver import Feature 
from torch import nn 
from gpuutils import GpuUtils

def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a set of nodes.
    """
    # assert(nfeat.device == torch.device('cpu')
    if type(nfeat) == Feature:
        batch_inputs = nfeat[input_nodes]
    else:    
        batch_inputs = nfeat[input_nodes.to('cpu')].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels



def train(rank, args, model, train_dataloader,\
             optimizer, features, labels, in_feat_dim, valid_dataloader, last_node_stored = 0):
    device = rank 
    E = [torch.cuda.Event(enable_timing = True) for i in range(5)]
    experiment_metrics = ExperimentMetrics()
    loss_fcn = nn.CrossEntropyLoss()
    
    for epoch in range(args.num_epochs):
        tic = time.time()
        model.train()
        E[0].record()
        epoch_start = time.time()
        epoch_metrics = EpochMetrics()
        logging.info("sample, movement, forward, backward, total_time, accuracy")
        for batch_id, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
            E[1].record()
            batch_start = time.time()
            minibatch_metrics = MiniBatchMetrics()
            if batch_id == 5:
                torch.cuda.profiler.start()
            if batch_id == 10:
                torch.cuda.profiler.stop()
            optimizer.zero_grad()
            batch_inputs, batch_labels = load_subtensor(
                features, labels, seeds, input_nodes, device)
            edges = []
            n = [blocks[0].num_src_nodes()]
            edges_computed = 0
            for blk in blocks:
                edges_computed += blk.edges()[0].shape[0]
                edges.append(blk.edges()[0].shape[0])
                n.append(blk.num_dst_nodes())
            minibatch_metrics.edges_computed = edges_computed    
            if last_node_stored != 0:
                hit = torch.sum(features.feature_order[input_nodes] < last_node_stored) 
            else: 
                hit = 0
            missed = input_nodes.shape[0] - hit
            minibatch_metrics.cpu_movement = missed * in_feat_dim * 4
            minibatch_metrics.gpu_movement = hit * in_feat_dim * 4
    
            E[2].record()
            # Compute loss and prediction
            torch.cuda.nvtx.range_push("training {} {}".format(edges,n))
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels.flatten())
            E[3].record()
            loss.backward()
            E[4].record()
            E[4].synchronize()
            torch.cuda.nvtx.range_pop()
            optimizer.step()
            minibatch_metrics.sample_time = E[0].elapsed_time(E[1])/1000
            minibatch_metrics.data_movement_time = E[1].elapsed_time(E[2])/1000
            minibatch_metrics.forward_time = E[2].elapsed_time(E[3])/1000
            minibatch_metrics.backward_time = E[3].elapsed_time(E[4])/1000
            minibatch_metrics.total_time = time.time() - batch_start 
            total_correct = batch_pred.argmax(dim=-1).eq(batch_labels).sum().item()/batch_pred.shape[0]
            minibatch_metrics.accuracy = total_correct
            epoch_metrics.add(minibatch_metrics)
            logging.info(f"{minibatch_metrics}")
            E[0].record()

        epoch_end = time.time()
        if rank == 0:
            with torch.no_grad():
                total_correct = 0
                total_predicted = 0
                for input_nodes, seeds, blocks  in valid_dataloader:
                    batch_inputs, batch_labels = load_subtensor(
                        features, labels, seeds, input_nodes, device)
                    batch_pred = model(blocks, batch_inputs)
                    total_correct += batch_pred.argmax(dim=-1).eq(batch_labels).sum().item()
                    total_predicted += seeds.shape[0]
            df = GpuUtils.analyzeSystem()
            memory_used = max(df['memory_usage_percentage'])        
            experiment_metrics.add(epoch_metrics, total_correct/total_predicted, epoch_end - epoch_start, memory_used)
            logging.info(f"epoch {epoch}, validation accuracy {total_correct/total_predicted}")
            torch.cuda.empty_cache()
        else: 
            experiment_metrics.add(epoch_metrics, 0, epoch_end - epoch_start, 0 )
            
    if rank == 0:
        experiment_metrics.compute_time()
    experiment_metrics.compute_volume()
    if rank == 0:
        logging.info(str(experiment_metrics))
    