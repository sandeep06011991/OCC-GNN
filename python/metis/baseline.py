import dgl
import numpy as np
import torch
from  ogb.nodeproppred import DglNodePropPredDataset
import shutil

DATA_DIR = "/data/sandeep"

def read_partition_file(graphname):
    DATA_DIR = "/data/sandeep"
    # graphname = "ogbn-products"
    # graphname = "ogbn-arxiv"
    TARGET_DIR = "{}/{}".format(DATA_DIR,graphname)
    #p_map = np.loadtxt(TARGET_DIR + "/metis.graph.part.4")
    p_map = np.loadtxt("dummy.metis.part.4")
    p_map = torch.from_numpy(p_map)
    return p_map

def read_graph(graphname):
    ROOT_DIR = "/data/sandeep"
    dataset = DglNodePropPredDataset(graphname, root = ROOT_DIR)
    graph = dataset[0][0]
    train_ids = dataset.get_idx_split()['train']
    return train_ids, graph

def layer_neighbour_sample(graph, layer_nds, cross_partition, pmap,  mask):
    l1 = []
    for nd in layer_nds:
        src_partition = pmap[nd]
        nbs_size = graph.in_degree(nd)
        if nbs_size == 0:
            continue
        nbs_id = torch.randint(nbs_size.item(),(20,))
        nbs,_ = graph.in_edges(nd)
        nbs = nbs[nbs_id]
        m1 = pmap[nbs]
        dest_partition = torch.unique(m1)
        v =  torch.sum(dest_partition != src_partition)
        cross_partition += v
        #dest_partition.shape[0]
        #mask[nd] += v
        mask[nd] += v
        #dest_partition.shape[0]
        #dest_partition.shape[0]
        l1.append(torch.unique(nbs))
    l1 = torch.cat(l1, dim = 0)
    l1 = torch.unique(l1)
    return l1, cross_partition

# Three hop simulation and print properties. 
def three_hop_simulation(training_nodes, graph, pmap):
    batch_size = 1024
    avg = []
    mask = torch.zeros(graph.num_nodes())
    training_node_balance = {}
    for i in range(4):
        training_node_balance[i] = torch.sum(pmap[training_nodes] == i)
    #print("Training Node Balance", training_node_balance)
    pure_nodes = 0
    memory_usage = torch.zeros(graph.num_nodes())
    comm_trace = torch.zeros(graph.num_nodes())
    for i in range(0,len(training_nodes),batch_size):
        minibatch = training_nodes[i:i+batch_size]
        cross_partition = 0
        layer = minibatch
        for k in range(2):
            layer, cross_partition = layer_neighbour_sample(graph, layer,  cross_partition, pmap,  mask)
            comm_trace[layer] += 1
            if k==1:
                memory_usage[layer] += 1
        avg.append(cross_partition)
    #torch.save(comm_trace, "{}/{}/comm_trace".format(DATA_DIR, "ogbn-arxiv"))
    print(mask, torch.sum(mask), torch.max(mask))
    for i in range(4):
        print("For partition",i)
        nds = torch.where(pmap == i)[0]
        s = torch.argsort(mask[nds], descending = True, dim = 0)
        e = torch.argsort(memory_usage[nds], descending = True, dim = 0)
        print("Cross partition comm: Max",torch.max(mask[nds]),"Mean:", torch.mean(mask[nds]) ,\
                "Mean top 100:", torch.mean(mask[nds[s[:100]]]))
        print("Memory Usage: Max", torch.max(memory_usage[nds]), \
                        "Mean:", torch.mean(memory_usage[nds]), "Mean top 100:", torch.mean(memory_usage[nds[e[:100]]]))
        print("training_nodes",training_node_balance[i])
        print("partition_size",nds.shape[0])
    print("Average cross partition", sum(avg)/len(avg))
    print("Checking for bias")
    #torch.save(mask,"comm_vol")
    mask_order = torch.argsort(mask, descending = True)
    one_hop_comm = torch.zeros(graph.num_nodes())
    total_degree = torch.zeros(graph.num_nodes())
    for i in range(graph.num_nodes()):
        src,_= graph.in_edges(i)
        _,dest = graph.out_edges(i)
        total_degree[i] = src.shape[0] + dest.shape[0]
        one_hop_comm[i] += torch.sum(torch.unique(pmap[torch.cat([src,dest],dim = 0)])!=pmap[i])
    # Mask order contains actual computation, one hop is what algorihtm worked on.
    print("Mask", mask[mask_order[:10]])
    print("One hop", one_hop_comm[:10])
    for i in range(4):
        rank = (torch.where(one_hop_comm[mask_order]==i)[0])
        print("rank of ", i)
        print("Communication rank", rank[:10])
        print("Node ids", mask_order[rank[:10]])
        print("Actual communication",mask[mask_order[rank[:10]]])
        print("Actualdegree",total_degree[mask_order[rank[:10]]],"##")
        print("Frequency",comm_trace[mask_order[rank[:10]]], "###") 
    return
    #torch.save(mask,"ogbn-mark.pt")
    #
    s = torch.argsort(mask, descending = True, dim = 0) 
    #print("max comm", mask[s[:30]])
    print("Properties of max",torch.max(mask), torch.min(mask), torch.mean(mask), torch.median(mask))
    #print("max degree", graph.in_degree(s[:10]), graph.out_degrees(s[:10]))
    #print("node ids", s[:30])
    #for i in s[:30]:
    #    s,d = graph.in_edges(i)
    #    print("node in edges", i, pmap[s])
    #    print("node out edges", i, pmap[graph.in_edges(i)[1]])
    print("Average cross partition", sum(avg)/len(avg))
    print("purity", pure_nodes, graph.num_nodes())


def write_metis_file_unbiased_edgecut(dg_graph,graphname):
    dg_graph = dgl.to_homogeneous(dg_graph)
    dg_graph = dgl.to_bidirected(dg_graph)
    print(dg_graph)
    mat = dg_graph.adj(scipy_fmt = 'csr')
    degree_mat = mat.indptr[1:]-mat.indptr[:-1]
    print("Max degree calculated is ", max(degree_mat))
    degree_mat = (10 * degree_mat / max(degree_mat)).astype(int)
    num_nodes = dg_graph.num_nodes()
    print(mat.indices.shape[0])
    # assert(num_edges == mat.indices.shape[0])
    ne = dg_graph.num_edges()/2
    with open("{}/{}/unbiased".format(DATA_DIR,graphname),'w') as fp:
        # fp.write("{} {} {}\n".format(num_nodes,int(dg_graph.num_edges()   /2),100))
        fp.write("{} {} {}\n".format(num_nodes,int(dg_graph.num_edges()   /2),000))
        for i in range(num_nodes):
            edges = mat.indices[mat.indptr[i]:mat.indptr[i+1]]+1
            edges_str = [str(ee) for ee in edges]
            line = " ".join(edges_str)
            # fp.write("{} {}\n".format(degree_mat[i] ,line))
            fp.write("{}\n".format(line))

def write_metis_file_weighted_train_nodes(dg_graph,graphname, train_nodes):
    dg_graph = dgl.to_homogeneous(dg_graph)
    dg_graph = dgl.to_bidirected(dg_graph)
    print(dg_graph)
    mask = torch.zeros(dg_graph.num_nodes())
    mask[train_nodes] = 1
    mat = dg_graph.adj(scipy_fmt = 'csr')
    degree_mat = mat.indptr[1:]-mat.indptr[:-1]
    print("Max degree calculated is ", max(degree_mat))
    degree_mat = (10 * degree_mat / max(degree_mat)).astype(int)
    num_nodes = dg_graph.num_nodes()
    print(mat.indices.shape[0])
    # assert(num_edges == mat.indices.shape[0])
    ne = dg_graph.num_edges()/2
    with open("{}/{}/train-biased".format(DATA_DIR,graphname),'w') as fp:
        # fp.write("{} {} {}\n".format(num_nodes,int(dg_graph.num_edges()   /2),100))
        fp.write("{} {} {}\n".format(num_nodes,int(dg_graph.num_edges()   /2),'010'))
        for i in range(num_nodes):
            edges = mat.indices[mat.indptr[i]:mat.indptr[i+1]]+1
            edges_str = [str(ee) for ee in edges]
            line = " ".join(edges_str)
            # fp.write("{} {}\n".format(degree_mat[i] ,line))
           
            fp.write("{} {}\n".format(str(int(mask[i])), line))


def run_metis_obj_cut(graphname, fileformat, isvol):
    import subprocess

    if isvol:
        output = subprocess.run(["/home/spolisetty/metis-5.1.0\
/build/Linux-x86_64/programs/gpmetis",\
                    "{}/{}/{}".format(DATA_DIR,graphname, fileformat),"4" \
                    ,"-objtype=vol"],
                    capture_output = True)
    else:
        output = subprocess.run(["/home/spolisetty/metis-5.1.0\
/build/Linux-x86_64/programs/gpmetis",\
                    "{}/{}/{}".format(DATA_DIR,graphname, fileformat),"4"],\
                    capture_output = True)
    if isvol:
        ext_string="vol"
    else:
        ext_string="cut"

    print(output.stderr)
    assert(len(output.stderr.strip()) == 0)
    output = str(output.stdout)
    print("metis", output)
    filelocation = "{}/{}/{}".format(DATA_DIR,graphname, fileformat)
    shutil.move("{}.part.4".format(filelocation), "{}-{}.part.4".format(filelocation,ext_string))
    #read_partition_file(graphname)

def read_partition_file(graphname, fileformat):
    p_map = np.loadtxt(DATA_DIR + "/{}/{}.part.4".format(graphname, fileformat))
    return torch.from_numpy(p_map)



if __name__ == "__main__":
    graphname = 'ogbn-arxiv'
    train_ids, graph = read_graph(graphname)
    #write_metis_file_weighted_train_nodes(graph,graphname, train_ids)
    #write_metis_file_unbiased_edgecut(graph,graphname)
    #Options for arg2: train-biased, unbiased
    #run_metis_obj_cut(graphname,"unbiased",True)
    p_map = read_partition_file(graphname,"comm-biased")
    three_hop_simulation(train_ids, graph, p_map)
