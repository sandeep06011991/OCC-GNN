# Reaches around 0.7870 ± 0.0036 test accuracy.
import os

import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import time
import argparse
from torch_geometric.nn import GATConv

####################
# Import Quiver
####################
import quiver


def average(ls):
    
    if(len(ls) == 1):
        return ls[0]
    return sum(ls[1:])/(len(ls)-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(SAGE, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all, device, subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, num_layers):
        super(GAT,self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads))
        for _ in range(num_layers -2):
            self.convs.append(GATConv(hidden_channels *heads, hidden_channels, heads))
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)
    
    def inference(self, x_all, device, subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


def run(rank, world_size, args,  quiver_sampler, quiver_feature, y, edge_index, split_idx, num_features, num_classes):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '11111'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    device_list = [0,1,2,3]
    torch.cuda.set_device(device_list[rank])

    train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']

    train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]
    print("Training nodes",train_idx.shape)
    train_loader = torch.utils.data.DataLoader(train_idx, batch_size=args.batch_size, pin_memory=True)

    if rank == 0:
        subgraph_loader = NeighborSampler(edge_index, node_idx=None,
                                          sizes=[-1], batch_size=args.batch_size,
                                          shuffle=False, num_workers=6)

    torch.manual_seed(12345)
    device_id = device_list[rank]
    heads = args.num_hidden 
    if args.model == "GAT":
        model = GAT(num_features, args.num_hidden, num_classes, heads,  num_layers = 3).to(device_id)    
    else:
        assert(args.model == "GCN")
        model = SAGE(num_features, args.num_hidden, num_classes, num_layers=3).to(device_id)
    model = DistributedDataParallel(model, device_ids=[device_id])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    y = y.to(device_id)
    print("start training")

    epoch_time = []
    sample_time_epoch = []
    movement_time_graph_epoch = []
    movement_time_feature_epoch = []
    forward_time_epoch = []
    backward_time_epoch = []
    accuracy = []
    e0 = torch.cuda.Event(enable_timing = True)
    e1 = torch.cuda.Event(enable_timing = True)
    e2 = torch.cuda.Event(enable_timing = True)
    e3 = torch.cuda.Event(enable_timing = True)
    e4 = torch.cuda.Event(enable_timing = True)
    for epoch in range(0, 2):
        model.train()
        epoch_start = time.time()
        sample_time = 0
        movement_graph_time = 0
        movement_feature_time = 0
        forward_time = 0
        backward_time = 0
        for seeds in train_loader:
            #print(seeds)
            e0.record()
            t1 =time.time()
            n_id, batch_size, adjs = quiver_sampler.sample(seeds)
            t2 = time.time()
            adjs = [adj.to(device_id) for adj in adjs]
            optimizer.zero_grad()
            t3 = time.time()
            e1.record()
            f = quiver_feature[n_id]
            e2.record()
            #torch.cuda.synchronize()
            t4 = time.time()
            out = model(f, adjs)
            e3.record() 
            t5 = time.time()
            loss = F.nll_loss(out, y[n_id[:batch_size]])
            loss.backward()
            e4.record()
            e4.synchronize()
            t6 = time.time()
            sample_time += (t2-t1)
            movement_graph_time += (t3-t2)
            movement_feature_time += max((t4-t3),e1.elapsed_time(e2)/1000)
            forward_time += max((t5-t4),e2.elapsed_time(e3)/1000)
            backward_time += max((t6 - t5, e3.elapsed_time(e4)/1000))
            optimizer.step()

        dist.barrier()
        epoch_time.append(time.time()-epoch_start)
        sample_time_epoch.append(sample_time)
        movement_time_graph_epoch.append(movement_graph_time)
        movement_time_feature_epoch.append(movement_feature_time)
        forward_time_epoch.append(forward_time)
        backward_time_epoch.append(backward_time)
        if rank == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {time.time() - epoch_start}')

        if rank == 0 and epoch % 5 == 0:  # We evaluate on a single GPU for now
            model.eval()
            with torch.no_grad():
                out = model.module.inference(quiver_feature, rank, subgraph_loader)
            res = out.argmax(dim=-1) == y.cpu()
            acc1 = int(res[train_idx].sum()) / train_idx.numel()
            acc2 = int(res[val_idx].sum()) / val_idx.numel()
            acc3 = int(res[test_idx].sum()) / test_idx.numel()
            accuracy.append(acc1)
            print(f'Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}')

        dist.barrier()
    if rank == 0:
        print("accuracy:", accuracy[-1])
        print("epoch:",average(epoch_time))
        print("sample_time:",average(sample_time_epoch))
        print("movement graph:",average(movement_time_graph_epoch))
        print("movement feature:",average(movement_time_feature_epoch))
        print("forward time:",average(forward_time_epoch))
        print("backward time:",average(backward_time_epoch))
    dist.destroy_process_group()


if __name__ == '__main__':
    root = "/mnt/bigdata/sandeep/"
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--graph',type = str, default= "ogbn-arxiv")
    argparser.add_argument('--cache-per', type =float, default = .25)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--batch-size', type=int, default=(1032))
    argparser.add_argument('--model', type = str)
    
    args = argparser.parse_args()
    assert(args.model in ["GCN","GAT"])
    dataset = PygNodePropPredDataset(args.graph, root)
    data = dataset[0]
    num_nodes = data.x.shape[0]
    split_idx = dataset.get_idx_split()
    split_idx["train"] = torch.where(torch.rand((num_nodes,)) < .80)[0]
    world_size = torch.cuda.device_count()
    world_size = 4
    ##############################
    # Create Sampler And Feature
    ##############################
    csr_topo = quiver.CSRTopo(data.edge_index)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [10, 10, 10], 0, mode="GPU")
    feature = torch.zeros(data.x.shape)
    feature[:] = data.x
    print("Total Node Feature Shape",feature.shape)
    #quiver.init_p2p(device_list = list(range(world_size)))
    quiver.init_p2p(device_list = [0,1,2,3])
    device_cache_size = args.cache_per * feature.shape[1] * num_nodes * 4 / (1032 * 1032)
    quiver_feature = quiver.Feature(rank=0, \
            device_list = [0,1,2,3], \
            #device_list=list(range(world_size)),\
            device_cache_size = "{}M".format(int(device_cache_size)),
            #device_cache_size = "400M",\
            #device_cache_size="2G", \
            cache_policy = "p2p_clique_replicate",\
            #cache_policy="device_replicate", 
                csr_topo=None)
    # csr_topo is none to prevent reordering
    quiver_feature.from_cpu_tensor(feature)

    print('Let\'s use', world_size, 'GPUs!')
    mp.spawn(
        run,
        args=(world_size, args, quiver_sampler, quiver_feature, data.y.squeeze(), data.edge_index, split_idx, dataset.num_features, dataset.num_classes),
        nprocs=world_size,
        join=True
    )
