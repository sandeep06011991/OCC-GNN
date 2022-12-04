from utils.utils import get_process_graph
from cslicer import cslicer
from models.dist_gcn import get_sage_distributed
from data.part_sample import *
from data.serialize import *
import torch

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    false_labels = torch.where(torch.argmax(pred,dim = 1) != labels)[0]
    return (torch.argmax(pred, dim=1) == labels).float().sum(),len(pred )


class TestAccuracy:

    def __init__(self, test_filename, device, self_edge):
        filename = test_filename
        fsize = -1
        ##### Read all data
        dg_graph, _, num_classes = get_process_graph(filename, fsize, True)
        num_nodes = dg_graph.ndata['features'].shape[0]
        deterministic = True
        gpu_local_storage = [i for i in range(num_nodes)]
        storage_vector = [gpu_local_storage,[],[],[]]
        fanout = -1 # is irrelevant
        testing = True
        rounds = 3
        pull_optimization = False
        num_layers = 3
        self.self_edge = self_edge
        self.sampler = cslicer(filename, storage_vector, fanout, deterministic, testing, self_edge, rounds, pull_optimization, num_layers)
        test_mask = dg_graph.ndata['test_mask']
        self.test_ids = torch.where(test_mask)[0]

        device = torch.device(device)
        self.device = device
        # read local test data
        ## Create dummy model
        hidden = 16

        features = dg_graph.ndata["features"].to(device)
        num_nodes = features.shape[0]

        fsize = features.shape[0]
        self.labels = dg_graph.ndata["labels"].to(device)
        gpu_id = torch.device(0)
        num_class = torch.max(self.labels) + 1
        # model = get_sage_distributed(hidden, features, num_classes, gpu_id, deterministic, model)
        ######## Dummy model
        ####### Generate Sample into correct format
        # # iterate through graph without sampling
        # # Test GCN
        self.features = features.to(device)


    def test_accuracy(self, model , sample_nodes):
        assert(False)
        # modify in features properly with cache hits and miss
        sample_nodes = sample_nodes.tolist()
        print("Sample Nodes",sample_nodes)
        if len(sample_nodes) <= 0:
            return
        csample = self.sampler.getSample(sample_nodes)
        tensorized_sample = Sample(csample)
        obj = Gpu_Local_Sample()
        obj.set_from_global_sample(tensorized_sample,0)
        data = serialize_to_tensor(obj)
        data = data.to(self.device)
        gpu_local_sample = Gpu_Local_Sample()
        # print(data.device , device)
        construct_from_tensor_on_gpu(data, self.device, gpu_local_sample)
        gpu_local_sample.prepare()
        with torch.no_grad():
            in_features = self.features[gpu_local_sample.cache_miss_from]
            predicted = model.forward(gpu_local_sample, in_features, True)
            return predicted

    def get_accuracy(self, model):
        batch_size = 1024 * 4
        num_test = self.test_ids.shape[0]
        test = 0
        correct = 0
        print("start testing")
        with torch.no_grad():
            for i in range(0, num_test, batch_size):
                sample_nodes = self.test_ids[i:i + batch_size].tolist()
                # print(sample_nodes)
                if len(sample_nodes) <= 0:
                    continue
                csample = self.sampler.getSample(sample_nodes)
                tensorized_sample = Sample(csample)
                obj = Gpu_Local_Sample()
                obj.set_from_global_sample(tensorized_sample,0)
                data = serialize_to_tensor(obj)
                data = data.to(self.device)
                gpu_local_sample = Gpu_Local_Sample()
                # print(data.device , device)
                construct_from_tensor_on_gpu(data, self.device, gpu_local_sample)
                gpu_local_sample.prepare(attention = self.self_edge)
                in_feat = torch.empty(gpu_local_sample.cache_hit_to.shape[0], *self.features.shape[1:], device = 0) 
                in_feat[gpu_local_sample.cache_hit_to] = self.features[gpu_local_sample.cache_hit_from]
                predicted = model.forward(gpu_local_sample, in_feat ,True)
                # print(predicted)
                correct_labels = self.labels[gpu_local_sample.out_nodes]
                # print(correct)
                a, b = compute_acc(predicted, correct_labels)

                correct = correct + a.item()
                test = test + b
                print(i,num_test)
            assert(test == num_test)
        return correct/test

if __name__=="__main__":
    filename = "ogbn-arxiv"
    fsize = -1
    ##### Read all data
    dg_graph, _, num_classes = get_process_graph(filename, fsize)
    num_nodes = dg_graph.ndata['features'].shape[0]
    deterministic = True
    fanout = 20 # This is irrelevant as deterministic true
    gpu_local_storage = [i for i in range(num_nodes)]
    storage_vector = [gpu_local_storage,[],[],[]]
    sampler = cslicer(filename, storage_vector, fanout, deterministic, True)
    test_mask = dg_graph.ndata['test_mask']
    test_ids = torch.where(test_mask)[0]
    device = torch.device(0)
    # read local test data
    ## Create dummy model
    features = dg_graph.ndata["features"].to(device)
    fsize = features.shape[0]
    labels = dg_graph.ndata["labels"]
    gpu_id = torch.device(0)
    model = "gcn"
    num_class = torch.max(labels) + 1
    model = get_sage_distributed(hidden, features, num_classes, gpu_id, deterministic, model)
    ######## Dummy model
    ####### Generate Sample into correct format
    batch_size = 1024
    sample_nodes = test_ids[:batch_size].tolist()
    # print(sample_nodes)
    csample = sampler.getSample(sample_nodes)
    tensorized_sample = Sample(csample)
    obj = Gpu_Local_Sample()
    obj.set_from_global_sample(tensorized_sample,0)
    data = serialize_to_tensor(obj)
    data = data.to(device)
    gpu_local_sample = Gpu_Local_Sample()
    print(data.device , device)
    construct_from_tensor_on_gpu(data, device, gpu_local_sample)
    gpu_local_sample.prepare()
    # # iterate through graph without sampling
    # # Test GCN
    features = features.to(device)
    model = model.to(device)
    with torch.no_grad():
        out = model.forward(gpu_local_sample, features, None, True)
    print(out)
    print("Forward pass ok!")
    # Test GAT Model
    # # Create a model and run it through data to validate
