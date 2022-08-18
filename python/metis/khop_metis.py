import ogb
import dgl
from  ogb.nodeproppred import DglNodePropPredDataset
import torch
import dgl.function as fn
ROOT_DIR = "/data/sandeep"
import numpy as np

#File to explore neighbour sample based graph construction for partition
#Cons: To far off from existing infrastructure

def partition_original_graph():
    pass

def reorder_layer(graph, target_nodes, node_prob, layer_offset):
    N = graph.num_nodes()
    mask = torch.zeros(N,dtype = torch.bool)
    mask[train_ids] = 1
    src,dest = graph.edges()
    edges_id = torch.where(mask[dest]==1)[0]
    d = torch.unique(dest[edges_id])
    s = torch.unique(src[edges_id])
    reorder_v = torch.zeros(N,dtype = torch.long)
    reorder_v[d] = 1 + torch.arange(d.shape[0],dtype = torch.long)
    assert(torch.max(reorder_v) == d.shape[0])
    reorder_u = torch.zeros(N,dtype = torch.long)
    reorder_u[s] = torch.arange(s.shape[0], dtype = torch.long) + d.shape[0] + 1
    print("edge_ids",edges_id.shape)
    bp_graph = dgl.heterograph({('_U','_E','_V'):(reorder_u[src[edges_id]],reorder_v[dest[edges_id]])})
    #propogate in degree
    bp_rev =bp_graph.reverse()
    bp_rev.nodes['_V'].data['in'] = 1/bp_graph.in_degrees()
    bp_rev.update_all(fn.copy_u('in', 'm'), fn.sum('m', 'out'))
    prob = bp_rev.nodes['_U'].data['out']
    #calculate probability of weight of lower node
    #convert heterograph to bi-directed 
    num_u = bp_graph.num_nodes('_U')
    num_v = bp_graph.num_nodes('_V')
    print(num_u,num_v, d.shape[0],s.shape[0])

def reorder_for_metis():
    dataset = DglNodePropPredDataset('ogbn-arxiv', root = ROOT_DIR)
    graph = dataset[0][0]
    train_ids = dataset.get_idx_split()['train']
    
            

def reorder():
    # Get original graph and train data splits
    dataset = DglNodePropPredDataset('ogbn-arxiv', root = ROOT_DIR)
    graph = dataset[0][0]
    train_ids = dataset.get_idx_split()['train']
    # Create bipartite graph with node renumbering 
        # select edges of train_ids
    N = graph.num_nodes()
    mask = torch.zeros(N,dtype = torch.bool)
    mask[train_ids] = 1
    src,dest = graph.edges()
    edges_id = torch.where(mask[dest]==1)[0]
    d = torch.unique(dest[edges_id])
    s = torch.unique(src[edges_id])
    reorder_v = torch.zeros(N,dtype = torch.long)
    reorder_v[d] = 1 + torch.arange(d.shape[0],dtype = torch.long)
    assert(torch.max(reorder_v) == d.shape[0])
    reorder_u = torch.zeros(N,dtype = torch.long)
    reorder_u[s] = torch.arange(s.shape[0], dtype = torch.long) + d.shape[0] + 1
    print("edge_ids",edges_id.shape)
    bp_graph = dgl.heterograph({('_U','_E','_V'):(reorder_u[src[edges_id]],reorder_v[dest[edges_id]])})
    #propogate in degree
    bp_rev =bp_graph.reverse()
    bp_rev.nodes['_V'].data['in'] = 1/bp_graph.in_degrees()
    bp_rev.update_all(fn.copy_u('in', 'm'), fn.sum('m', 'out'))
    prob = bp_rev.nodes['_U'].data['out']
    #calculate probability of weight of lower node
    #convert heterograph to bi-directed 
    num_u = bp_graph.num_nodes('_U')
    num_v = bp_graph.num_nodes('_V')
    print(num_u,num_v, d.shape[0],s.shape[0])
    # First layer edges num_u then second layer offseted nodes num_u + num_v
    with open('dummy.metis','w') as fp:
        fp.write("{} {} {}\n".format(num_u - 1, bp_graph.num_edges(),str(100)))
        count = 0
        for i in range(1,num_v):
            src, dest = bp_graph.in_edges(i)
            count += dest.shape[0]
            ls = [str(i) for i in src.tolist()]
            ls = " ".join(ls)
            fp.write("{} {}\n".format(str(0),ls))
        for i in range(num_v,num_u):
            src, dest = bp_graph.out_edges(i)
            count += dest.shape[0]
            ls = [str(i) for i in dest.tolist()]
            ls = " ".join(ls)
            fp.write("{} {}\n".format(int(prob[i] * 100),ls))
            #fp.write("{} {}\n".format(str(1),ls))
    print("total writte", count)
    print("file written")    
    return reorder_u, reorder_v, graph ,train_ids       

    
def run_one_hop_simulation(graph, training_nodes, s, d, pmap, batch_size):
    avg = []
    graph.in_edges(
    for i in range(0,len(training_nodes),batch_size):
        minibatch = training_nodes[i:i+batch_size]    
        cross_partition = 0
        for nd in minibatch:
            src_partition = pmap[d[nd]]
            nbs_size = graph.in_degree(nd)
            if(nbs_size==0):
                continue
            nbs_id = torch.randint(nbs_size.item(),(20,))
            nbs,_ = graph.in_edges(nd)
            nbs = nbs[nbs_id]
            m1 = pmap[s[nbs]]
            print(nd,m1)
            dest_partition = torch.unique(m1)
            cross_partition +=  torch.sum(dest_partition != src_partition)
        avg.append(cross_partition)
    print("Average cross partition", sum(avg)/len(avg))



def run_metis():
    # graphname = "ogbn-arxiv"
    DATA_DIR = "/data/sandeep"
    import subprocess
    output = subprocess.run(["/home/spolisetty/metis-5.1.0\
/build/Linux-x86_64/programs/gpmetis",\
                    "dummy.metis","4","-objtype=vol"],
                    capture_output = True)
    print(output.stderr)
    assert(len(output.stderr.strip()) == 0)
    output = str(output.stdout)
    print("metis", output)

def read_partition_file():
    DATA_DIR = "/data/sandeep"
    # graphname = "ogbn-products"
    # graphname = "ogbn-arxiv"
    p_map = np.loadtxt("dummy.metis.part.4")
    pmap = torch.from_numpy(p_map)
    nmap = torch.zeros(pmap.shape[0]+1)
    nmap[1:] = pmap
    return nmap






if __name__ == "__main__":
    #experiment1
    s,d,graph,training_ids = reorder()
    run_metis()
    p_map = read_partition_file()
    run_one_hop_simulation(graph, training_ids, s,d,p_map, 1024)
