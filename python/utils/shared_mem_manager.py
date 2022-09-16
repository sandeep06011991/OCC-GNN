SHARED_MEMORY_SIZE = 10 * 1024 * 1024
NUM_BUCKETS = 4
from multiprocessing.shared_memory import SharedMemory
import multiprocessing as mp
import numpy as np
import torch
import time
# Everything outside this class should be agnostic to shared memory details
# Keep this object in memory and delete after everything returns
class SharedMemManager():

    def __init__(self, free_memory_filenames):
        self.buckets = {}
        self.free_memory_filenames = free_memory_filenames
        for i in range(NUM_BUCKETS):
            name = 's{}'.format(i)
            self.buckets[name] = SharedMemory(name,create = True, size =SHARED_MEMORY_SIZE)
            print("Created", name)
            free_memory_filenames.put(name)

    def __del__(self):
        print("Shared memory manager should wait for all allocated memory to be returned")
        print("Check if queue is completely clear to implement.")
        for k in self.buckets.keys():
            self.buckets[k].close()
            self.buckets[k].unlink()
        # Clear all shared memory

class SharedMemClient():

    def __init__(self, free_memory_filenames):
        self.buckets = {}
        self.free_memory_filenames = free_memory_filenames
        for i in range(NUM_BUCKETS):
            name = 's{}'.format(i)
            self.buckets[name] = SharedMemory(name, size =SHARED_MEMORY_SIZE)
        self.used_memory = {}

    def write_to_shared_memory(self ,data):
        # returns filename written to
        assert(type(data) == np.ndarray)
        assert(len(data.shape) == 1)
        assert(data.shape[0] * data.dtype.itemsize < SHARED_MEMORY_SIZE)
        while(True):
            try:
                name = self.free_memory_filenames.get_nowait()
                break
            except:
                print("Allocate more shared memory")
                time.sleep(10)
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
        del self.used_memory[filename]

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

    sm_files = mp.Queue(NUM_BUCKETS)
    meta_exchange = mp.Queue(NUM_BUCKETS)
    mg = SharedMemManager(sm_files)
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
