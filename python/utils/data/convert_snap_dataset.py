import sys
import dgl
import torch
import numpy as np
import subprocess
from env import get_root_dir
# All one time preprocessing goes here.
ROOT_DIR = get_root_dir()
TARGET_DIR = get_root_dir()

def runcmd(cmd, verbose = False, *args, **kwargs):
    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass
# Todos:
# Add steps to see if the file exists
# Else pull them, and unzip.
def get_dataset(name):
    if name == "com-orkut":
        runcmd("wget -P download_folder https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz", verbose = False)
        runcmd("gzip -d download_folder https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz", verbose = False)
        > /somewhere/file
        edgelist = np.loadtxt("{}/{}/com-orkut.ungraph.txt".\
            format(ROOT_DIR,"com-orkut"),dtype = int)
    return edgelist

def write_dataset_dataset(edgelist):
    # dataset = dgl.add_self_loop(dataset)
    target = TARGET_DIR + "/" + name
    import os
    os.makedirs(target,exist_ok = True)
    edges = torch.from_numpy(edgelist)
    print(edges.shape)
    graph = dgl.DGLGraph((edges[:,0],edges[:,1]))
    graph.remove_self_loop()
    sparse_mat = graph.adj(scipy_fmt='csr')
    sparse_mat.sort_indices()
    assert(np.array_equal(np.ones(sparse_mat.data.shape),sparse_mat.data))
    indptr = sparse_mat.indptr
    indices = sparse_mat.indices
    print("offset",indptr.sum())
    print("edges", indices.sum())
    num_edges = graph.num_edges()
    num_nodes = graph.num_nodes()

    csum_offsets = indptr.sum()
    csum_edges = indices.sum()

    assert indptr.shape == (num_nodes+1,)
    assert indices.shape == (num_edges,)

    meta = {}

    with open(target + '/indptr.bin','wb') as fp:
        fp.write(indptr.astype(np.int64).tobytes())
    with open(target + '/indices.bin','wb') as fp:
        fp.write(indices.astype(np.int64).tobytes())

    meta_structure = {}
    meta_structure["num_nodes"] = num_nodes
    meta_structure["num_edges"] = num_edges
    meta_structure["csum_offsets"] = csum_offsets
    meta_structure["csum_edges"] = csum_edges

    with open(target + '/meta.txt','w') as fp:
        for k in meta_structure.keys():
            fp.write("{}={}\n".format(k,meta_structure[k]))
    print("All data written!")

if __name__=="__main__":
    # assert(len(sys.argv) == 3)
    name = "com-orkut"
    edgelist = get_dataset(name)
    write_dataset_dataset(edgelist)
    print("max nodes",np.max(edgelist))
    print("Snap preprocessing  done !")
