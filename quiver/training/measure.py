from dataclasses import dataclass
from dataclasses import field 


def avg(ls):
    if(len(ls) == 1):
        return ls[0]
    if (len(ls) < 3):
        return sum(ls)/2
    a = max(ls[1:])
    b = min(ls[1:])
    return (sum(ls[1:]) -a -b)/(len(ls)-3)

@dataclass(init= False)
class MiniBatchMetrics:
    sample_time : float
    data_movement_time : float 
    forward_time : float 
    backward_time : float 
    total_time : float 
    cpu_movement: int
    gpu_movement: int 
    edges_computed: int 
    accuracy: float
    def __repr__(self) -> str:
        return f'{self.sample_time:.2f},' + \
            f'{self.data_movement_time:.2f},' + \
                f'{self.forward_time:.2f},' + \
                    f'{self.backward_time:.2f},' + \
                        f'{self.total_time:.2f},{self.accuracy:.2f}'

@dataclass
class EpochMetrics:
    sample_time: int = 0
    data_movement_time : int = 0
    forward_time : int = 0 
    backward_time : int = 0 
    total_time : int = 0 
    cpu_movement: int = 0
    gpu_movement: int = 0 
    edges_computed: int = 0

    def add(self, batch: MiniBatchMetrics):
        self.sample_time += batch.sample_time 
        self.data_movement_time += batch.data_movement_time
        self.forward_time += batch.forward_time
        self.backward_time += batch.backward_time
        self.total_time += batch.total_time 
        self.cpu_movement += batch.cpu_movement
        self.gpu_movement += batch.gpu_movement 
        self.edges_computed += batch.edges_computed 

@dataclass
class ExperimentMetrics:
    sample_time: list = field(default_factory=list)
    data_movement_time : list[float] = field(default_factory=list)
    forward_time : list[float] = field(default_factory=list)
    backward_time : list[float] = field(default_factory=list)
    total_time : list[float] = field(default_factory=list)
    cpu_movement: list[float] = field(default_factory=list)
    gpu_movement: list = field(default_factory=list)     
    valid_accuracy: list[float] = field(default_factory=list)
    edges: list[float] = field(default_factory=list)
    percentage_memory_used: list[float] = field(default_factory=list)

    def add(self, epoch:EpochMetrics, valid_accuracy: float, total_time, percentage_memory):
        self.sample_time.append(epoch.sample_time)
        self.data_movement_time.append(epoch.data_movement_time)
        self.forward_time.append(epoch.forward_time)
        self.backward_time.append(epoch.backward_time)
        self.cpu_movement.append(epoch.cpu_movement)
        self.gpu_movement.append(epoch.gpu_movement)
        self.valid_accuracy.append(valid_accuracy)
        self.edges.append(epoch.edges_computed)
        self.total_time.append(total_time)
        self.percentage_memory_used.append(percentage_memory)

    def compute_volume(self):
        print(f"gpu_data_moved:{avg(self.gpu_movement)/(1024 ** 3)}GB")
        print(f"cpu_data_moved:{avg(self.cpu_movement)/(1024 ** 3)}GB")
        print(f"edges_per_epoch:{avg(self.edges)}")
    
    def compute_time(self):
        print(f"accuracy:{avg(self.valid_accuracy):.4f}")
        print(f"epoch_time:{avg(self.total_time):.4f}")
        print(f"sample_time:{avg(self.sample_time):.4f}")
        print(f"movement_feature:{avg(self.data_movement_time):.4f}")
        print(f"forward_time:{avg(self.forward_time):.4f}")
        print(f"backward_time:{avg(self.backward_time):.4f}")
        print(f"percentage gpu memory used:{max(self.percentage_memory_used):.4f}")
    def __repr__(self) -> str:
        s = ''
        s += f"accuracy:{avg(self.valid_accuracy):.4f}"
        s += f"epoch_time:{avg(self.total_time):.4f}"
        s += f"sample_time:{avg(self.sample_time):.4f}"
        s += f"movement_feature:{avg(self.data_movement_time):.4f}"
        s += f"forward_time:{avg(self.forward_time):.4f}"
        s += f"backward_time:{avg(self.backward_time):.4f}"
        s += f"percentage gpu memory used:{max(self.percentage_memory_used):.4f}"
        return s