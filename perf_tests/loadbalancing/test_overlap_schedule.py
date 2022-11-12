# Not automated test
from utils.utils import get_process_graph
import dgl
import torch
import dgl.function as fn
import time

def sync_schedule():
    graph,_,_ = get_process_graph("ogbn-products",-1)
    features = graph.ndata["features"]
    N = graph.num_nodes()
    cuda = features.to(0)
    features = features.pin_memory()
    u,v = graph.edges()
    ht_g = dgl.heterograph({('_V','_E','_U'):(u,v)},{'_U':N,'_V':N})
    ht_g = ht_g.reverse()
    ht_g = ht_g.to(0)

    ht_g.nodes['_U'].data['in'] = features.to(0)
    e1 = torch.cuda.Event(enable_timing = True)
    e2 = torch.cuda.Event(enable_timing = True)
    for i in range(10):
        t1 = time.time()
        e1.record()
        features.to(0, non_blocking = True)
        ht_g.update_all(fn.copy_u('in', 'm'), fn.sum('m', 'out'))

        e2.record()
        e2.synchronize()
        t2 = time.time()
        print("Time", t2-t1, "Event time", e1.elapsed_time(e2)/1000)

def async_schedule():
    graph,_,_ = get_process_graph("ogbn-products",-1)
    features = graph.ndata["features"]
    cuda = features.to(0)
    N = graph.num_nodes()
    features = features.pin_memory()
    u,v = graph.edges()
    ht_g = dgl.heterograph({('_V','_E','_U'):(u,v)},{'_U':N,'_V':N})
    ht_g = ht_g.reverse()
    ht_g = ht_g.to(0)

    ht_g.nodes['_U'].data['in'] = features.to(0)
    e1 = torch.cuda.Event(enable_timing = True)
    e2 = torch.cuda.Event(enable_timing = True)
    e3 = torch.cuda.Event(enable_timing = True)
    e4 = torch.cuda.Event(enable_timing = True)
    e5 = torch.cuda.Event(enable_timing = True)
    e6 = torch.cuda.Event(enable_timing = True)
    s1 = torch.cuda.Stream(0)
    s2 = torch.cuda.Stream(0)
    features = features.to(0)
    for i in range(10):
        t1 = time.time()
        e1.record()
        with torch.cuda.stream(s1):
            e3.record()
            features[:10000,:].to(1, non_blocking = True)
            e4.record()
        with torch.cuda.stream(s2):
            e5.record()
            ht_g.update_all(fn.copy_u('in', 'm'), fn.sum('m', 'out'))
            e6.record()
        s1.synchronize()
        s2.synchronize()
        t2 = time.time()

        e2.record()
        e2.synchronize()

        print("Time", t2-t1, "Event time", e1.elapsed_time(e2)/1000)
        print("Move time", e3.elapsed_time(e4)/1000)
        print("BAndwirdht", features[:1000000,:].shape[0] * 100 * 4 /(1024 * 1024 * 1024)/(e3.elapsed_time(e4)/1000))
        print("comp time", e5.elapsed_time(e6)/1000)

if __name__=="__main__":
    async_schedule()
