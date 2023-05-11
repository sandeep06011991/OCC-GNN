import torch
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
import dgl
import os
from os.path import exists
from metis import *

# Since this is the easiest dataset to generateself.
# Allow overwriting.
# file_exists = exists(path_to_file)
import dgl
import time
import torch
'''
@Input: 
1. N: Number of parititions
2. Graph: A dgl graph
@ Output: 
1. VID_TO_PARTITION: Mapping of original id to parition id
'''
def partition(Graph: dgl.DGLGraph, N: int) -> torch.Tensor:
    VID_TO_PARTITION = dgl.metis_partition_assignment(Graph, N)
    return VID_TO_PARTITION
    
'''
@Input: 
1. VID_TO_PARTITION: Original id to parition id
2. N: Number of parititions
3. Graph: Dgl graph, if specified sort new id by degree

@ Output: 
1. ORG_TO_NEW: Orignal id to new id
Note: 
# The Vertex ID is contiguous in each partition. 
# The Vertex ID is ordered in increasing degree. (ID_1 > ID_2 <-> Deg(ID_1) >= Deg(ID_2))
2. start: [MinID in Part1,   MinID Part2,...] in each partition
3. end:   [MaxID-1 in Part1, MaxID-1 in Part2,...] in each partition
'''
def org_to_newid(VID_TO_PARTITION: torch.Tensor, Graph: dgl.DGLGraph, N: int):
    start = time.time()
    num_vertex = VID_TO_PARTITION.shape[0]
    start = [0] * N
    end = [0] * N
    ORG_TO_NEW = torch.empty(num_vertex, dtype=torch.int64)
    cur_idx = 0
    
    # map original vertex ids to contiguous id chunks
    for partition in range(0, N):
        start[partition] = cur_idx # start of local index
        for idx, cur_partition in enumerate(VID_TO_PARTITION):
            if cur_partition == partition:
                ORG_TO_NEW[idx] = cur_idx
                cur_idx += 1
        end[partition] = cur_idx # first index greater than the maximum of local index
        print(f"{partition=} {start=} {end=} num_vertex={end[partition] - start[partition]}")
    
    if Graph is None:
        return ORG_TO_NEW, start, end
    
    # sort contigous id by in_degree 
    # small id -> small in_degree   
    def comp(vid):
        return Graph.in_degrees(vid)
    
    for i in range(0, N):
        cur_start = start[i]
        cur_end = end[i]
        org_ids = [] 
        for org_id in range(0, num_vertex):
            new_id = ORG_TO_NEW[org_id]
            if (new_id >= cur_start and new_id < cur_end):
                org_ids.append(new_id)

        org_ids.sort(key = comp)
        cur_idx = cur_start
        prev_deg = 0
        
        for org_id in org_ids:
            ORG_TO_NEW[org_id] = cur_idx
            cur_idx += 1

    return ORG_TO_NEW, start, end

def get_pmap(start:list[int], end: list[int], num_vertex: int) -> torch.Tensor:
    def get_parition_id(vid) -> int:
        for i in range(len(start)):
            if vid < start[i]:
                return i - 1
        return len(start) - 1
    
    out = torch.empty(num_vertex, dtype=torch.int32)
    for i in range(num_vertex):
        out[i] = get_parition_id(i)
    return out

def map_edges(vid_to_newid: torch.Tensor, Graph: dgl.DGLGraph):
    org_in, org_out = Graph.edges()
    new_src = torch.clone(org_in)
    new_dst = torch.clone(org_out)
    for i in range(Graph.num_nodes()):
        org_in_id  = org_in[i]
        org_out_id = org_out[i]
        new_src[i]  = vid_to_newid[org_in_id]
        new_dst[i] = vid_to_newid[org_out_id]
    print(f"max_id={torch.max(vid_to_newid)} max_src={torch.max(new_src)} max_dst={torch.max(new_dst)}")
    return new_src, new_dst

def map_feature(vid_to_newid: torch.Tensor, org_feat: torch.Tensor):
    new_feat = torch.clone(org_feat)
    for org_id, new_id in enumerate(vid_to_newid):
        new_feat[new_id] = org_feat[org_id]
    return new_feat

def map_labels(vid_to_newid: torch.Tensor, labels: torch.Tensor):
    new_labels = torch.clone(labels)
    for org_id, new_id in enumerate(vid_to_newid):
        new_labels[new_id] = labels[org_id]
    return new_labels

def RangeReindex(Graph: dgl.DGLGraph, labels:torch.Tensor, N: int):
    # edges = Graph.edges()
    # Graph.remove_edges(torch.where(edges[0] == edges[1])[0])
    org_feat = Graph.ndata["feat"]
    vid_to_partition = partition(Graph, N)
    vid_to_newid, start, end = org_to_newid(vid_to_partition, Graph, N)
    num_nodes = len(vid_to_newid)
    p_map = get_pmap(start, end, num_nodes)
    feat = map_feature(vid_to_newid, org_feat)
    print(f"{num_nodes=} {feat.shape=}")
    src_ids, dst_ids = map_edges(vid_to_newid, Graph)
    g = dgl.graph((src_ids, dst_ids))
    g.ndata['feat'] = feat
    g.ndata['p_map'] = p_map
    new_labels = map_labels(vid_to_newid, labels)
    
    return g, new_labels, vid_to_newid

def get_data_dir():
    import os
    username = os.environ['USER']
    if username == 'spolisetty_umass_edu':
        DATA_DIR = "/work/spolisetty_umass_edu/data"
    if username == "spolisetty":
        DATA_DIR = "/data/sandeep"
    if username == "q91":
        DATA_DIR = "/mnt/bigdata/sandeep"
    if username == "juelin":
        DATA_DIR = "/home/ubuntu/dataset"
    return DATA_DIR

ROOT_DIR = get_data_dir()

def write_dataset_dataset(name, TARGET_DIR, N:int=4):
    # DGL graphs area always direction src to edges
    dataset = DglNodePropPredDataset(name, root=ROOT_DIR)
    graph, labels = dataset[0]
    # edges = graph.edges()
    # dgl graph edges are always src to destination.
    # graph.remove_edges(torch.where(edges[0] == edges[1])[0])
    graph, labels, vid_to_newid = RangeReindex(graph, labels, N)
    sparse_mat = graph.adj(scipy_fmt='csr')
    sparse_mat.sort_indices()
    assert(np.array_equal(np.ones(sparse_mat.data.shape), sparse_mat.data))
    indptr = sparse_mat.indptr
    indices = sparse_mat.indices
    # c slicer should be processing from
    # dgl graphs are in src to dest
    # indptr is for srcs
    # However sampling must start from dest
    # Thus reverse this.
    c_spmat = graph.adj(scipy_fmt = 'csr', transpose = True)
    c_indptr = c_spmat.indptr
    c_indices = c_spmat.indices

    print(indptr.shape)
    print(indices.shape)
    print("offset", indptr.sum())
    print("edges", indices.sum())

    num_edges = graph.num_edges()
    num_nodes = graph.num_nodes()
    nan_lab = torch.where(torch.isnan(labels.flatten()))[0]
    labels = labels.flatten()
    labels[nan_lab] = 0

    print("Lables", torch.max(labels))

    num_classes = int(torch.max(labels) + 1)
    features = graph.ndata['feat']
    assert features.shape[0] == num_nodes
    assert labels.shape[0] == num_nodes
    feature_dim = features.shape[1]
    csum_features = torch.sum(features).item()
    csum_labels = torch.sum(labels).item()
    csum_offsets = indptr.sum()
    csum_edges = indices.sum()
    
    assert indptr.shape == (num_nodes+1,)
    assert indices.shape == (num_edges,)
    if('train_idx' not in graph.ndata.keys()):
        split_idx = dataset.get_idx_split()
        train_idx = split_idx['train']
        val_idx = split_idx['valid']
        test_idx = split_idx['test']
        
        train_idx = vid_to_newid[train_idx]
        val_idx = vid_to_newid[val_idx]
        test_idx = vid_to_newid[test_idx]
        print("Train", train_idx)
        print("Val", val_idx)
        print("Test", test_idx)
        
    p_map = graph.ndata['p_map']
    with open(TARGET_DIR + '/partition_map.bin','wb') as fp:
        fp.write(p_map.numpy().astype(np.int32).tobytes())
    with open(TARGET_DIR+'/cindptr.bin', 'wb') as fp:
        fp.write(c_indptr.astype(np.int64).tobytes())
    with open(TARGET_DIR+'/cindices.bin', 'wb') as fp:
        fp.write(c_indices.astype(np.int64).tobytes())
    with open(TARGET_DIR+'/indptr.bin', 'wb') as fp:
        fp.write(indptr.astype(np.int64).tobytes())
    with open(TARGET_DIR+'/indices.bin', 'wb') as fp:
        fp.write(indices.astype(np.int64).tobytes())
    with open(TARGET_DIR+'/features.bin', 'wb') as fp:
        fp.write(features.numpy().astype('float32').tobytes())
    with open(TARGET_DIR+'/labels.bin', 'wb') as fp:
        fp.write(labels.numpy().astype('int32').tobytes())
    with open(TARGET_DIR+'/train_idx.bin', 'wb') as fp:
        fp.write(train_idx.numpy().astype('int64').tobytes())
    with open(TARGET_DIR+'/val_idx.bin', 'wb') as fp:
        fp.write(val_idx.numpy().astype('int64').tobytes())
    with open(TARGET_DIR+'/test_idx.bin', 'wb') as fp:
        fp.write(test_idx.numpy().astype('int64').tobytes())
    csum_train = torch.sum(train_idx).item()
    csum_test = torch.sum(val_idx).item()

    meta_structure = {}
    meta_structure["num_nodes"] = num_nodes
    meta_structure["num_edges"] = num_edges
    meta_structure["feature_dim"] = feature_dim
    meta_structure["csum_features"] = csum_features
    meta_structure["csum_labels"] = csum_labels
    meta_structure["csum_train"] = csum_train
    meta_structure["csum_test"] = csum_test
    meta_structure["csum_offsets"] = csum_offsets
    meta_structure["csum_edges"] = csum_edges
    meta_structure["num_classes"] = num_classes
    with open(TARGET_DIR+'/meta.txt', 'w') as fp:
        for k in meta_structure.keys():
            fp.write("{}={}\n".format(k, meta_structure[k]))
            
# arg0 = dgl dataset name
# arg1 = full target directory
if __name__=="__main__":
    # assert(len(sys.argv) == 3)
    nname = ["ogbn-products", "ogbn-arxiv"]
    # Note papers 100M must be reordered
    for name in nname:
        target = ROOT_DIR + "/" + name + "-metis"
        os.makedirs(target, exist_ok=True)
        write_dataset_dataset(name, target)
