# How to use cslicer in pythonic way.

import torch
from cslicer import cslicer
from utils import memory_manager
from utils.utils import get_process_graph
from utils.memory_manager import MemoryManager
def get_total_comm(s):
    x = 0
    y = 0
    for i in range(4):
        y = y + s.missing_node_ids[i].shape[0]
    for id,l in enumerate(s.layers):
        for l_id,bp in enumerate(l):
            for f_id, f in enumerate(bp.from_ids):
                x = x + f.shape[0]
                # print(id, l_id, f_id, f.shape[0])
    return x, y

batch_size = 4096
graph = "ogbn-products"
fsize = -1
dg_graph, partition_map, num_classes = get_process_graph(graph, fsize, testing = False)
features = dg_graph.ndata['features']
train_mask = dg_graph.ndata['train_mask']
train_nids = torch.where(train_mask)[0]
import random
train_nids = train_nids.tolist()
random.shuffle(train_nids)

Last Layer nodes
tensor([163653,  59728, 134956, 115062,  30994], device='cuda:0')
Last Layer nodes
tensor([ 42384,  66332, 106385,  61545,  90419], device='cuda:1')
Last Layer nodes
tensor([     0, 164033,  67034,  28932, 105878], device='cuda:3')
Last Layer nodes
tensor([119004, 164689,   9646,  72946,  93857], device='cuda:2')
tensor(532467., device='cuda:1', grad_fn=<SumBackward0>) layer gpu sum
tensor(600878., device='cuda:2', grad_fn=<SumBackward0>) layer gpu sum
tensor(1247861., device='cuda:0', grad_fn=<SumBackward0>) layer gpu sum
tensor(649984., device='cuda:3', grad_fn=<SumBackward0>) layer gpu sum
tensor(499171., device='cuda:0', grad_fn=<SumBackward0>) layer gpu sum
tensor(421807., device='cuda:1', grad_fn=<SumBackward0>) layer gpu sum
tensor(555719., device='cuda:3', grad_fn=<SumBackward0>) layer gpu sum
tensor(412980., device='cuda:2', grad_fn=<SumBackward0>) layer gpu sum
tensor(66370., device='cuda:0', grad_fn=<SumBackward0>) layer gpu sum
tensor(56822., device='cuda:1', grad_fn=<SumBackward0>) layer gpu sum
tensor(173633., device='cuda:3', grad_fn=<SumBackward0>) layer gpu sum
Expected value 66370.0 303229509
tensor(66283., device='cuda:2', grad_fn=<SumBackward0>) layer gpu sum
Expected value 56822.0 275928863
Expected value 173633.0 267938497
Expected value 66283.0 257971338
Expected value tensor(66370., device='cuda:0', grad_fn=<SumBackward0>) 303229509
Expected value tensor(56822., device='cuda:1', grad_fn=<SumBackward0>) 275928863
Expected value tensor(173633., device='cuda:3', grad_fn=<SumBackward0>) 267938497
Expected value tensor(66283., device='cuda:2', grad_fn=<SumBackward0>) 257971338

Read graph with number of nodes: 169343
debugdummy_storage_map size15501:142156 163653 59728 134956 115062
dummy_storage_map size11610:150344 42384 66332 106385 61545
dummy_storage_map size16009:72253 119004 164689 9646 72946
dummy_storage_map size19011:52991 0 164033 67034 28932
[2022-10-18 20:48:00.111] [info] Ignore the perf results as this is wrong
layer bp sum33374
layer bp sum116575
layer bp sum106857
layer bp sum173114
layer bp sum83888
layer bp sum292238
layer bp sum495208
layer bp sum224539
layer bp sum1214886
layer bp sum586982
layer bp sum114972
layer bp sum492972
My anser is 1105068207sample_val 1105068207


cache_pers = [0,.25,1]
m_dict = {}
slicer_dict = {}
fanout = [10,10,10]
testing = False
deterministic = False
for cache in cache_pers:
    m_dict[cache] = MemoryManager(dg_graph, features, dg_graph.ndata["labels"],\
                     cache, \
                        fanout, batch_size, partition_map, deterministic = False)
    storage_vector = []
    for i in range(4):
        storage_vector.append(m_dict[cache].local_to_global_id[i].tolist())
    slicer_dict[cache] = cslicer(graph, storage_vector, 10, True, False)

# // Get this from data
for i in range(0, len(train_nids),batch_size):
    training_nodes = train_nids[i:i + batch_size]
    d = {}
    for cache in cache_pers:
        s1 = slicer_dict[cache].getSample(training_nodes)
        a = get_total_comm(s1)
        d[cache] = a
    print(d)
