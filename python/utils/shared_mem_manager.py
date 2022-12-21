SHARED_MEMORY_SIZE = 300 * 1024 * 1024
# Number of workers * 2 = Quesize = num buckets.
# NUM_BUCKETS = 4
from multiprocessing.shared_memory import SharedMemory
import multiprocessing as mp
import numpy as np
import torch
import time
from utils.log import *

def get_number_buckets(num_workers):
    print("change to make sampling not a bottleneck")
    # previously num_workers * 2 * 4 * num_workers
    return num_workers * 2 * num_workers 
# Everything outside this class should be agnostic to shared memory details
# Keep this object in memory and delete after everything returns
class SharedMemManager():

    def __init__(self, free_memory_filenames, num_workers, file_id):
        self.buckets = {}
        self.free_memory_filenames = free_memory_filenames
        NUM_BUCKETS = get_number_buckets(num_workers)
        print("Creating buckets", NUM_BUCKETS)
        for i in range(NUM_BUCKETS):
            name = '{}_aa{}'.format(file_id,i)
            self.buckets[name] = SharedMemory(name,create = True, size =SHARED_MEMORY_SIZE)
            free_memory_filenames.put(name)
        print("Created buckets", NUM_BUCKETS)

    def __del__(self):
        print("Shared memory manager should wait for all allocated memory to be returned")
        print("Check if queue is completely clear to implement.")
        for k in self.buckets.keys():
            self.buckets[k].close()
            self.buckets[k].unlink()
        # Clear all shared memory

class SharedMemClient():

    def __init__(self, free_memory_filenames, name, worker_id, num_workers, file_id):
        self.buckets = {}
        self.free_memory_filenames = free_memory_filenames
        NUM_BUCKETS = get_number_buckets(num_workers)
        for i in range(NUM_BUCKETS):
            name = '{}_aa{}'.format(file_id, i)
            self.buckets[name] = SharedMemory(name, size =SHARED_MEMORY_SIZE)
        self.used_memory = {}
        self.log = LogFile(name, worker_id)

    def write_to_shared_memory(self ,data):
        # returns filename written to
        assert(type(data) == np.ndarray)
        assert(len(data.shape) == 1)
        if(data.shape[0] * data.dtype.itemsize > SHARED_MEMORY_SIZE):
            print("Shared Memory bucket size is not enouch for {}".format(data.shape[0] * data.dtype.itemsize))
            self.log.log("Shared Memory bucket size is not enouch for {}".format(data.shape[0] * data.dtype.itemsize))
        assert(data.shape[0] * data.dtype.itemsize < SHARED_MEMORY_SIZE)
        self.log.log("Trying to write, avail shared memory buckets {}".format(self.free_memory_filenames.qsize()))
        #print("shared memory queue", self.free_memory_filenames.qsize())
        name = self.free_memory_filenames.get()

        # while(True):
            # try:
            #     name = self.free_memory_filenames.get_nowait()
            #     break
            # except:
            #     self.log("")
            #     print("Allocate more shared memory")
            #     time.sleep(.001)
        buff = self.buckets[name]
        # SharedMemory(name = name , size = SHARED_MEMORY_SIZE)
        allocate = np.ndarray(*data.shape,dtype = data.dtype,  buffer = buff.buf)
        allocate[:] = data[:]
        # buff.close()
        return name

    def read_from_shared_memory(self, filename, shape, dtype):
        buff = self.buckets[filename]
        read_array = np.ndarray(*shape, dtype = dtype , buffer = buff.buf)
        tensor = torch.from_numpy(read_array)
        self.used_memory[filename] = True
        return tensor

    def free_used_shared_memory(self,filename):
        # Dont free shared memoryself.
        # Keep it always open
        self.free_memory_filenames.put(filename)
        self.used_memory[filename] = False

    def __del__(self):
        for name in self.buckets.keys():
            self.buckets[name].close()

def unit_test_for_correctness():
    def producer(sm_file, meta_exchange):
        data = np.random.rand(1000)
        sm_client = SharedMemClient(sm_file)
        name = sm_client.write_to_shared_memory(data)
        meta_exchange.put((name, data.shape, data.dtype.name))
        print("Write sum",np.sum(data))

    def consumer(sm_file, meta_exchange):
        sm_client = SharedMemClient(sm_file)
        name, shape, str_dtype = meta_exchange.get()
        print("Meta ", name, shape)
        dtype = np.dtype(str_dtype)
        tensor = sm_client.read_from_shared_memory(name, shape, dtype)
        print("Read sum", torch.sum(tensor))
    NUM_BUCKETS = get_number_buckets(3)
    sm_files = mp.Queue(NUM_BUCKETS)
    meta_exchange = mp.Queue(NUM_BUCKETS)
    mg = SharedMemManager(sm_files,3)
    print("Current file size ", sm_files.qsize())
    p = mp.Process(target = producer, args = (sm_files, meta_exchange))
    p.start()
    p.join()
    p = mp.Process(target = consumer, args = (sm_files, meta_exchange))
    p.start()
    p.join()
    print("All data consumed !!")

if __name__ == "__main__":
    unit_test_for_correctness()
