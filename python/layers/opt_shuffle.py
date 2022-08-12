from torch.multiprocessing import Queue
import torch
from torch import multiprocessing as mp
import torch as th
from torch.nn.parallel import DistributedDataParallel
import time,datetime
import torch.cuda.nvtx  as nvtx
# FixMe: Currently blocking. Test with NVLink and asynchronous send/recv
class Shuffle(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_t, queues, device_id, to_dict, from_dict, layer_id):
        temp = []
        temp_g = []
        t1 = time.time()
        from_data = 0
        to_data = 0
        debug = False
        to_data_unique = 0
        from_data_unique = 0
        for i in range(4):
            if i == device_id:
                temp.append(None)
                temp_g.append(None)
            else:
                # data could be used for debugging
                from_data += from_dict[i].shape[0]
                to_data += to_dict[i].shape[0]
                to_data_unique += torch.unique(to_dict[i]).shape[0]
                from_data_unique += torch.unique(from_dict[i]).shape[0]
                temp.append(torch.empty((from_dict[i].shape[0], *input_t.shape[1:]) \
                    , device = device_id))
                temp_g.append(torch.empty((to_dict[i].shape[0], *input_t.shape[1:]) \
                    , device = device_id))
        if layer_id == 0:
            print("Sending traffic node", to_data , "UNIQUE", to_data_unique, "size in MB", to_data * input_t.shape[1] * 4 /(1024 * 1024), device_id)
            print("recieving traffic ",from_data , "UNIQUE", from_data_unique, "size in MB",from_data * input_t.shape[1] * 4 /(1024 * 1024), device_id)
        remote_obj = []
        torch.distributed.barrier(device_ids = [device_id])
        # Uses blocking commpunication
        t1 = time.time()
        send_time = 0
        recv_time = 0
        print("Start shuffle",device_id)
        # nvtx.range_push("send and recieve {}".format(device_id))
        for from_id in range(4):
            t111 = time.time()
            for to_id in range(4):
                if device_id == 0:
                    print("from id", from_id, to_id)
                if to_id == from_id:
                    continue
                if from_id == device_id and to_dict[to_id].shape[0] != 0:
                    t11 = time.time()
                    a = input_t[to_dict[to_id]].clone().detach()
                    # assert(not torch.all(a==0))
                    remote_obj.append(torch.distributed.send(a,to_id,tag = device_id))
                    send_time += (time.time() - t11)
                    if debug:
                        assert(not torch.all(a==0))
                        print("sending", from_id, to_id, torch.sum(a))
                if to_id == device_id and from_dict[from_id].shape[0] != 0:
                    t11 = time.time()
                    remote_obj.append(torch.distributed.recv(temp[from_id], src=from_id, tag=from_id))
                    recv_time += (time.time() - t11)
                    a = torch.sum(temp[from_id])
                    if debug:
                        print("recieving", from_id, to_id, torch.sum(temp[from_id]))
            t222 = time.time()
            # if to_id== device_id:
            #     print("total time to send",t222 - t111, "to id", device_id)
        # nvtx.range_pop()
        print("stuck at barrier",device_id)
        torch.distributed.barrier(device_ids = [device_id])
        t2 = time.time()
        # if layer_id == 0:
        #     print("shuffle time layer FINE GRAINED",t2- t1,device_id)
            # print("total sent and recieved", send_time, recv_time)

        # for obj in remote_obj:
        #     obj.wait()
        # torch.distributed.barrier()
        t3 = time.time()
        for from_id in range(4):
            if from_id == device_id or from_dict[from_id].shape[0] == 0:
                continue
            input_t[from_dict[from_id]] += temp[from_id]
        print("end shuffle",device_id)
        # torch.distributed.barrier()
        t4 = time.time()

        ctx.queues = queues
        ctx.device_id = device_id
        ctx.from_dict = from_dict
        ctx.to_dict = to_dict
        ctx.temp_g = temp_g
        ctx.grad_required = input_t.requires_grad
        ctx.layer_id = layer_id
        ctx.debug = debug
        if (layer_id ==0):
            print("Total time SHUFFLE TIME IN LAYER 0 ", (t4 - t3),t2- t1 )
        return input_t

    @staticmethod
    def backward(ctx, grad_output):
        debug = ctx.debug
        device_id = ctx.device_id
        send_queue = []
        queues = ctx.queues
        from_dict = ctx.from_dict
        to_dict = ctx.to_dict
        temp = ctx.temp_g
        layer_id = ctx.layer_id
        t1 = time.time()
        irecv_queue = []
        t3 = time.time()
        out = grad_output.clone()
        send_queue = []
        remote_obj = []
        torch.distributed.barrier(device_ids = [device_id])
        # assert(torch.any(out != 0))
        for to_id in range(4):
            for from_id in range(4):
                if to_id == from_id:
                    continue
                if from_id == device_id and to_dict[to_id].shape[0] != 0:
                    remote_obj.append(torch.distributed.recv(temp[to_id], src=to_id, tag= device_id))
                    if debug:
                        print("recieving", to_id, from_id, torch.sum(temp[to_id]))
                if to_id == device_id and from_dict[from_id].shape[0] != 0:
                    a = out[from_dict[from_id]].detach()
                    remote_obj.append(torch.distributed.send(a, from_id, tag=from_id))
                    if debug:
                        print("sending", to_id, from_id, torch.sum(a))

        torch.distributed.barrier(device_ids = [device_id])
        # for obj in remote_obj:
        #     obj.wait()


        for to_id in range(4):
            if to_id == device_id or to_dict[to_id].shape[0] == 0:
                continue
            out[to_dict[to_id]] +=  temp[to_id]
        return out,None,None, None, None, None


class ToySingle(torch.nn.Module):

    def __init__(self, queues, device_id):
        super(ToySingle, self).__init__()
        self.ll = torch.nn.Linear(100, 100)
        self.queues = queues
        self.device_id = device_id

    def forward(self, input, from_id, to_id):
        a = self.ll(input)
        b = Shuffle.apply(a, self.queues, self.device_id, to_id, from_id, 0)
        return b

def test_single(proc_id, n_gpus, queues):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    world_size = n_gpus
    pg = th.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id)
    from_id = {}
    to_id = {}
    for i in range(4):
        if i != proc_id:
            from_id[i] = torch.tensor(range(100)).to(proc_id)
            to_id[i] = torch.tensor(range(100)).to(proc_id)
    model = ToySingle(queues, proc_id).to(proc_id)
    model = DistributedDataParallel(model, device_ids = [proc_id])
    X_t = torch.rand((1000,100), device = proc_id)
    for i in range(1000):
        t1 = time.time()
        out = model.forward(X_t, from_id, to_id)
        t2 = time.time()
        out.sum().backward()
        t3 = time.time()
        print("Forward pass", proc_id, "time", t2-t1)
        print("Back pass", proc_id, "time", t3-t2)
    print("Forward pass working")

if __name__ == "__main__":
    # Create unit test which can handle  shuffling
    procs = []
    n_gpus = 4
    print("Launch multiple gpus")
    queues = [Queue() for i in range(n_gpus)]
    for proc_id in range(n_gpus):
        p = mp.Process(target=(test_single),
                       args=(proc_id, n_gpus, queues))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
