import torch
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import dgl
import tqdm


class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[:block.num_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.num_nodes(), self.n_hidden if l != len(
                self.layers) - 1 else self.n_classes).to(device)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.num_nodes(), device=g.device),
                sampler,
                device=device,
                batch_size=args.batch_size*4,
                shuffle=False,
                drop_last=False,
                num_workers=0 if args.sample_gpu else args.num_workers)

            for input_nodes, output_nodes, blocks in tqdm(dataloader):
                block = blocks[0].int().to(device)

                h = x[input_nodes].to(device)
                h_dst = h[:block.num_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h

            x = y
        return y

from ogb.nodeproppred import DglNodePropPredDataset

if __name__=="__main__":
    root = "/mnt/bigdata/sandeep/"
    data = DglNodePropPredDataset(name='ogbn-arxiv', root=root)
    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    graph, labels = data[0]
    block = dgl.to_block(graph)
    in_feats = 128
    n_hidden = 128
    torch.cuda.set_device(0)
    layer = dglnn.SAGEConv(in_feats, n_hidden, 'mean').to(0)
    block = block.to(0)
    f = graph.ndata['feat'].to(0)
    e1 = torch.cuda.Event(enable_timing = True)
    e2 = torch.cuda.Event(enable_timing = True)
    e3 = torch.cuda.Event(enable_timing = True)
    for i in range(10):
        e1.record()
        fp = layer(block, f)
        e2.record()
        fp.sum().backward()
        e3.record()
        e3.synchronize()
        print("fp",e1.elapsed_time(e2))
        print("bp",e2.elapsed_time(e3))
    print("All done")
