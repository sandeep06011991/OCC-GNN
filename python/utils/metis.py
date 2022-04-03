import numpy as np
import scipy.sparse
import numpy as np
import torch
from dgl import DGLGraph
import dgl

def create_metis_file(graphname):
    DATA_DIR = "/data/sandeep"
    # graphname = "ogbn-products"
    # graphname = "ogbn-arxiv"
    print("read file beging")
    indptr = np.fromfile("{}/{}/indptr.bin".format(DATA_DIR,graphname),dtype = np.int64)
    indices = np.fromfile("{}/{}/indices.bin".format(DATA_DIR,graphname),dtype = np.int64)
    num_nodes = indptr.shape[0] - 1
    num_edges = indices.shape[0]
    print("read file end")
    print("sp file read")
    sp = scipy.sparse.csr_matrix((np.ones(indices.shape),indices,indptr),
        shape = (num_nodes,num_nodes))
    print("sp file end")
    dg_graph = DGLGraph(sp)
    e = dg_graph.edges()
    assert(torch.all(e[0]!=e[1]))
    print(dg_graph)
    dg_graph = dgl.to_homogeneous(dg_graph)
    dg_graph = dgl.to_bidirected(dg_graph)
    print(dg_graph)
    mat = dg_graph.adj(scipy_fmt = 'csr')
    degree_mat = mat.indptr[1:]-mat.indptr[:-1]
    degree_mat = (10 * degree_mat / max(degree_mat)).astype(int)
    print(mat.indices.shape[0])
    # assert(num_edges == mat.indices.shape[0])
    ne = dg_graph.num_edges()/2
    with open("{}/{}/metis.graph".format(DATA_DIR,graphname),'w') as fp:
        fp.write("{} {} {}\n".format(num_nodes,int(dg_graph.num_edges()   /2),100))
        for i in range(num_nodes):
            edges = mat.indices[mat.indptr[i]:mat.indptr[i+1]]+1
            edges_str = [str(ee) for ee in edges]
            line = " ".join(edges_str)
            fp.write("{} {}\n".format(degree_mat[i] ,line))
    print("Write success!")
    # Read to file
    # Write to file
    return dg_graph
import torch

def read_partition_file(graphname):
    DATA_DIR = "/data/sandeep"
    # graphname = "ogbn-products"
    # graphname = "ogbn-arxiv"
    TARGET_DIR = "{}/{}".format(DATA_DIR,graphname)
    p_map = np.loadtxt(TARGET_DIR + "/metis.graph.part.4")
    print(p_map.shape)
    print(np.sum(p_map==0))
    indptr = np.fromfile("{}/{}/indptr.bin".format(DATA_DIR,graphname),dtype = np.int64)
    indices = np.fromfile("{}/{}/indices.bin".format(DATA_DIR,graphname),dtype = np.int64)
    num_nodes = indptr.shape[0] - 1
    num_edges = indices.shape[0]
    sp = scipy.sparse.csr_matrix((np.ones(indices.shape),indices,indptr),
        shape = (num_nodes,num_nodes))
    dg_graph = DGLGraph(sp)
    e = dg_graph.edges()
    pmap = torch.from_numpy(p_map)
    cross_edge = torch.where(pmap[e[0]]!=pmap[e[1]])[0]
    # print(e[0][cross_edge], e[1][cross_edge] )
    all_nds = torch.unique(torch.cat((e[0][cross_edge],e[1][cross_edge])))

    print("Halo Nodes",all_nds.shape[0]/num_nodes)
    with open(TARGET_DIR + '/partition_map_opt.bin','wb') as fp:
        fp.write(p_map.astype(np.intc).tobytes())
    print("Write done !!!")
    # Create a new partition file that can be read use this as a variable.
    pass

def run_metis(graphname):
    # graphname = "ogbn-arxiv"
    DATA_DIR = "/data/sandeep"
    dg_graph = create_metis_file(graphname)
    import subprocess
    output = subprocess.run(["/home/spolisetty/metis-5.1.0\
/build/Linux-x86_64/programs/gpmetis",\
                    "{}/{}/metis.graph".format(DATA_DIR,graphname),"4"],
                    capture_output = True)
    assert(len(output.stderr.strip()) == 0)
    output = str(output.stdout)
    # print("metis", output)
    read_partition_file(graphname)
