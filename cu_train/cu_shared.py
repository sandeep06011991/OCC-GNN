SHARED_MEMORY_SIZE = 100 * 1024 * 1024
from multiprocessing.shared_memory import SharedMemory
import multiprocessing as mp
import numpy as np
import torch
import time

#
def get_number_buckets(num_gpus):
    return num_gpus * num_gpus
# # Everything outside this class should be agnostic to shared memory details
# # Keep this object in memory and delete after everything returns
class SharedMemManager():

    def __init__(self, num_workers, file_id):
        self.buckets = {}
        # Each num worker refers to one gpu.
        NUM_BUCKETS = get_number_buckets(num_workers)
        print("Creating buckets", NUM_BUCKETS)
        for worker_id in range(num_workers):
            for sample_id in range(num_workers):
                name = '{}_worker_{}_sample_{}'.format(file_id, worker_id, sample_id)
                self.buckets[name] = SharedMemory(name,create = True, size =SHARED_MEMORY_SIZE)

    def __del__(self):
        print("Shared memory manager should wait for all allocated memory to be returned")
        print("Check if queue is completely clear to implement.")
        for k in self.buckets.keys():
            self.buckets[k].close()
            self.buckets[k].unlink()
        print("All done")
        # Clear all shared memory

class SharedMemClient():

    def __init__(self,  worker_id, num_workers, file_id):
        self.buckets = {}
        self.file_id = file_id
        print("Num worker ", num_workers)
        NUM_BUCKETS = get_number_buckets(num_workers)
        for worker_id in range(num_workers):
            for sample_id in range(num_workers):
                name = '{}_worker_{}_sample_{}'.format(file_id, worker_id, sample_id)
                print("Creating", name)
                self.buckets[name] = SharedMemory(name, size =SHARED_MEMORY_SIZE)
        self.used_memory = {}

    def write_to_shared_memory(self ,data, for_worker_id, sample_id):
        # returns filename written to
        assert(type(data) == np.ndarray)
        assert(len(data.shape) == 1)
        if(data.shape[0] * data.dtype.itemsize > SHARED_MEMORY_SIZE):
            print("Shared Memory bucket size is not enouch for {}".format(data.shape[0] * data.dtype.itemsize))
        assert(data.shape[0] * data.dtype.itemsize < SHARED_MEMORY_SIZE)
        name = '{}_worker_{}_sample_{}'.format(self.file_id, for_worker_id, sample_id)
        buff = self.buckets[name]
        # SharedMemory(name = name , size = SHARED_MEMORY_SIZE)
        allocate = np.ndarray(*data.shape,dtype = data.dtype,  buffer = buff.buf)
        allocate[:] = data[:]
        # buff.close()
        return name
    def read_from_shared_memory(self, worker_id, sample_id, shape, dtype):
        filename = '{}_worker_{}_sample_{}'.format(self.file_id, worker_id, sample_id)
        buff = self.buckets[filename]
        read_array = np.ndarray(*shape, dtype = dtype , buffer = buff.buf)
        tensor = torch.from_numpy(read_array)
        self.used_memory[filename] = True
        return tensor

    def free_used_shared_memory(self,filename):
#         # Dont free shared memoryself.
#         # Keep it always open
        self.used_memory[filename] = False
#
    def __del__(self):
        for name in self.buckets.keys():
            self.buckets[name].close()
