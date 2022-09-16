from data.serialize import *
from utils.shared_mem_manager import *
from data.bipartite import *
from data.part_sample import *
from cslicer import cslicer

def work_producer(work_queue, training_nodes, batch_size,
                no_epochs, num_workers,
                    deterministic):
    # todo:
    training_nodes = training_nodes.tolist()
    num_nodes = len(training_nodes)
    for epoch in range(no_epochs):
        i = 0
        if not deterministic:
            random.shuffle(training_nodes)
        while(i < num_nodes):
            work_queue.put(training_nodes[i:i+batch_size])
            i = i + batch_size
        work_queue.put("EPOCH")
    for n in range(num_workers):
        work_queue.put("END")
    while(True):
        if(work_queue.qsize()==0):
            break
        time.sleep(1)
    time.sleep(30)
    print("WORK PRODUCER TRIGGERING END")


# Work queue to get sample data.
# lock to write to 4 queues at the same time
# sample_queue = to put the meta result.
# storage vector to create samplers
# sm_filename_queue to co-ordinate access to shared memory
def slice_producer(graph_name, work_queue, sample_queues, \
    lock , storage_vector, \
        deterministic, worker_id, sm_filename_queue):
    no_worker_threads = 1
    sampler = cslicer(graph_name,storage_vector,10, deterministic)
    sm_client = SharedMemClient(sm_filename_queue)
    # Todo clean up unnecessary iterations
    while(True):
        sample_nodes = work_queue.get()
        if((sample_nodes) == "END"):
            lock.acquire()
            for qid,q in enumerate(sample_queues):
                sample_queues[qid].put("END")
            lock.release()
            #print("WORK SLICER RESPONDING TO END")
            break
        if sample_nodes == "EPOCH":
            lock.acquire()
            for qid,q in enumerate(sample_queues):
                sample_queues[qid].put("EPOCH")
            lock.release()
            continue
        csample = sampler.getSample(sample_nodes)
        tensorized_sample = Sample(csample)
        sample_id = tensorized_sample.randid
        gpu_local_samples = []
        dummy = []
        for gpu_id in range(4):
            # gpu_local_samples.append(Gpu_Local_Sample(tensorized_sample, gpu_id))
            obj = Gpu_Local_Sample()
            obj.set_from_global_sample(tensorized_sample,gpu_id)
            data = serialize_to_tensor(obj)
            data = data.numpy()
            print("Meta#############",data[:8])
            name = sm_client.write_to_shared_memory(data)
            ref = ((name, data.shape, data.dtype.name))
            gpu_local_samples.append(ref)
        # assert(False)
        #while(sample_queues[0].qsize() >= queue_size - 3):
        #    time.sleep(.01)
        lock.acquire()
        #print("ATTEMPT TO PUT SAMPLE",sample_queues[0].qsize(), "WORKER", worker_id)
        #print("Worker puts sample",sample_id)
        for qid,q in enumerate(sample_queues):
            while True:
                try:
                    sample_queues[qid].put_nowait(gpu_local_samples[qid])
                    print("Worker puts sample", sample_id,"in queue",qid)
                    break
                except:
                    time.sleep(.0001)
                    #print("Sampler is too fast, Exception putting stuff")
        lock.release()

    print("Waiting for sampler process to return")
    while(True):
        time.sleep(1)
        if(sample_queues[3].qsize()==0):
            break
    time.sleep(30)
    print("SAMPLER PROCESS RETURNS")
