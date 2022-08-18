import ogb
import dgl
from  ogb.nodeproppred import DglNodePropPredDataset
import torch
import dgl.function as fn
ROOT_DIR = "/data/sandeep"
import numpy as np

# Try different versions of bias
def biased_partition(graphname):
    # Get original graph and train data splits
    pm = torch.load('{}/{}/comm_trace'.format(ROOT_DIR,graphname))
    dataset = DglNodePropPredDataset(graphname, root = ROOT_DIR)
    graph = dataset[0][0]
    train_ids = dataset.get_idx_split()['train']
    # Create bipartite graph with node renumbering 
        # select edges of train_ids
    num_nodes = graph.num_nodes()
    graph = dgl.to_homogeneous(graph)
    graph = dgl.to_bidirected(graph)
    print(graph)
    mat = graph.adj(scipy_fmt = 'csr')
    degree_mat = mat.indptr[1:]-mat.indptr[:-1]
    print("Max degree calculated is ", max(degree_mat))
    degree_mat = (10 * degree_mat / max(degree_mat)).astype(int)
    is_train = torch.zeros(num_nodes,dtype = torch.bool)
    is_train[train_ids] = 1
    print(mat.indices.shape[0])
    # assert(num_edges == mat.indices.shape[0])
    ne = graph.num_edges()/2
    with open("{}/{}/comm-biased".format(ROOT_DIR, graphname),'w') as fp:
        # fp.write("{} {} {}\n".format(num_nodes,int(dg_graph.num_edges()   /2),100))
        fp.write("{} {} {}\n".format(num_nodes,int(graph.num_edges()   /2),110))
        for i in range(num_nodes):
            edges = mat.indices[mat.indptr[i]:mat.indptr[i+1]]+1
            edges_str = [str(ee) for ee in edges]
            line = " ".join(edges_str)
            # fp.write("{} {}\n".format(degree_mat[i] ,line))
            if is_train[i]:
                wt = "1"
            else:
                wt = "0"
                
            fp.write("{} {} {}\n".format(int(pm[i]),wt, line))

def run_metis():
    # graphname = "ogbn-arxiv"
    DATA_DIR = "/data/sandeep"
    import subprocess
    output = subprocess.run(["/home/spolisetty/metis-5.1.0\
/build/Linux-x86_64/programs/gpmetis",\
                    "{}/{}/comm-biased".format(DATA_DIR,graphname),"4","-objtype=vol"],
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
    return pmap






if __name__ == "__main__":
    #experiment1
    graphname = "ogbn-arxiv"
    biased_partition(graphname)
    run_metis()
    #p_map = read_partition_file()
