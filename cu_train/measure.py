from dataclasses import dataclass
from dataclasses import field 


def avg(ls):
    # assert(len(ls) > 3)
    print(ls)
    if len(ls) == 1:
        return ls[0]
    if(len(ls) <= 3):
        return sum(ls[1:])/len(ls[1:])
    a = max(ls[1:])
    b = min(ls[1:])
    # remove 3 as (remove first, max and min)
    return (sum(ls[1:]) - a - b)/(len(ls) - 3)

@dataclass(init = False)
class MinibatchMetrics:
    sample_get_time : float
    first_layer_time : float
    other_layer_time : float 
    forward_time : float 
    backward_time : float 
    movement_graph : float
    movement_feat : float 
    data_moved_per_gpu : int 
    data_moved_inter_gpu : int 
    edges_per_gpu : int 
    correct :int 
    predicted :int 

    


@dataclass
class EpochMetrics:
    sample_get_time : float = 0
    first_layer_time : float = 0
    other_layer_time : float  = 0
    forward_time : float = 0
    backward_time : float = 0
    movement_graph : float = 0
    movement_feat : float = 0
    data_moved_per_gpu : int = 0
    data_moved_inter_gpu : int = 0 
    edges_per_gpu : int = 0

    def append(self, minibatch):
        self.sample_get_time += minibatch.sample_get_time 
        # self.first_layer_time += minibatch.first_layer_time 
        # self.other_layer_time += minibatch.other_layer_time 
        self.forward_time += minibatch.forward_time 
        self.backward_time += minibatch.backward_time 
        self.movement_feat += minibatch.movement_feat 
        self.movement_graph += minibatch.movement_graph 
        self.data_moved_inter_gpu += minibatch.data_moved_per_gpu 
        self.data_moved_per_gpu += minibatch.data_moved_per_gpu 
        self.edges_per_gpu += minibatch.edges_per_gpu 


@dataclass
class ExperimentMetrics:
    sample_get_time: list = field(default_factory=list, repr = False)
    data_movement_time : list[float] = field(default_factory=list)
    forward_epoch : list[float] = field(default_factory=list)
    backward_epoch : list[float] = field(default_factory=list)
    movement_graph_epoch : list[float] = field(default_factory=list)
    movement_feat_epoch : list[float] = field(default_factory=list)
    epoch_time : list[float] = field(default_factory=list)
    epoch_accuracy: list[float] = field(default_factory=list)
    data_moved_per_gpu_epoch : list[float] = field(default_factory=list)
    data_moved_inter_gpu_epoch : list[float] = field(default_factory=list)
    edges_per_gpu_epoch : list[float] = field(default_factory=list)
    first_layer_time_epoch : list[float] = field(default_factory=list)
    other_layer_time_epoch: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory= list)
    epoch_time: list[float] = field(default_factory= list)

    def append(self, epoch, accuracy, epoch_time):
        self.accuracy.append(accuracy)
        self.sample_get_time.append(epoch.sample_get_time)
        self.forward_epoch.append(epoch.forward_time)
        self.backward_epoch.append(epoch.backward_time)
        self.movement_feat_epoch.append(epoch.movement_feat)
        self.movement_graph_epoch.append(epoch.movement_graph)
        self.data_moved_inter_gpu_epoch.append(epoch.data_moved_inter_gpu)
        self.data_moved_per_gpu_epoch.append(epoch.data_moved_per_gpu)
        self.epoch_time.append(epoch_time)
        
    def __str__(self) -> str:
                # Todo use the metrics data class 
        return f"\n\nVal accuracy: {max(self.accuracy):.3f}\n" + \
            f"Sample time: {avg(self.sample_get_time):.3f}\n" + \
            f"Epoch time: {avg(self.epoch_time):.3f}\n"
        # print("Test Accuracy:", self.test_accuracy_list)
        # print("accuracy:{}".format(acc))
        # print("#################",epoch_time)
        # print("epoch_time:{}".format(avg(epoch_time)))
        # print("sample_time:{}".format(avg(sample_get_epoch)))
        # print("movement graph:{}".format(avg(movement_graph_epoch)))
        # print("movement feature:{}".format(avg(movement_feat_epoch)))
        # print("forward time:{}".format(avg(forward_epoch)))
        # print("backward time:{}".format(avg(backward_epoch)))
        # print("data movement:{}MB".format(avg(data_moved_per_gpu_epoch)))
        # #print("Inter gpu data movement:{}MB".format(avg(data_moved_inter_gpu_epoch)))
        # print("Shuffle time:{}".format(avg(movement_partials_epoch)))
        # print("Memory Used:{} GB".format(torch.cuda.max_memory_allocated()/(1024 ** 3)))
        # print("First layer time:{}".format(avg(first_layer_time_epoch)))
        # print("other layer time:{}".format(avg(other_layer_time_epoch)))  
        # print("Exiting main training loop",sample_queue.qsize())
        # print("edges per epoch:{}".format(avg(edges_per_gpu_epoch)))
    