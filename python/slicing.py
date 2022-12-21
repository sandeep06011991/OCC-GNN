from data.serialize import *
from utils.shared_mem_manager import *
from data.bipartite import *
from data.part_sample import *
from cslicer import cslicer
from utils.log import *
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
            # Wierd bug on P4
            if(i + batch_size > num_nodes):
                break
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
# Slice producers only communicate with the first processesself.
# The first process communicates with other processes.
def slice_producer(graph_name, work_queue, sample_queue, \
    lock , storage_vector, \
        deterministic, worker_id, sm_filename_queue, num_workers, fanout,file_id, self_edge, pull_optimization, rounds):
    no_worker_threads = 1
    testing = False
    num_layers = 3
    sampler = cslicer(graph_name,storage_vector,fanout, deterministic, testing , self_edge, rounds, \
            pull_optimization, num_layers)
    sm_client = SharedMemClient(sm_filename_queue, "slicer", worker_id, num_workers,file_id)
    # Todo clean up unnecessary iterations
    #log = LogFile("slice-py", worker_id)
    while(True):
        sample_nodes = work_queue.get()
        # while True:
        #     try:
        #         sample_nodes = work_queue.get_nowait()
        #         print("Slicer got work @@@@@@@@@@@@@@@@@@@@@@@@@2")
        #         break
        #     except:
        #         print("Not able to get work !!!!!!!!!!!!")
        #         time.sleep(1)
        if((sample_nodes) == "END"):
            lock.acquire()
            sample_queue.put("END")
            lock.release()
            #print("WORK SLICER RESPONDING TO END")
            break
        if sample_nodes == "EPOCH":
            lock.acquire()
            sample_queue.put("EPOCH")
            lock.release()
            continue
        #log.log("ask cmodule for sample")
        # print("ask cmodule for sample")
        #print("#################################1")
        #print("Start sampling")
        csample = sampler.getSample(sample_nodes)
        #print("Sampling complete ")
        # print("cmodule returns sample, start tensorize")
        #log.log("cmodule returns sample, start tensorize")
        tensorized_sample = Sample(csample)
        # print("Finish tensorize")
        #log.log("Tensorization complete. start serialziation")
        sample_id = tensorized_sample.randid
        gpu_local_samples = []
            # print("Attemtong to serailize")
        for gpu_id in range(4):
            # gpu_local_samples.append(Gpu_Local_Sample(tensorized_sample, gpu_id))
            obj = Gpu_Local_Sample()
            obj.set_from_global_sample(tensorized_sample,gpu_id)
            data = serialize_to_tensor(obj)
            data = data.numpy()
            #print("Warning shared memory queue may be  emtpy ", sm_filename_queue.qsize())
            name = sm_client.write_to_shared_memory(data)
            ref = ((name, data.shape, data.dtype.name))
            gpu_local_samples.append(ref)
        #log.log("Serialization complete, write meta data to leader gpu process")
        # print("finish write and serialization.")
        # assert(False)
        #while(sample_queues[0].qsize() >= queue_size - 3):
        #    time.sleep(.01)
        #print("ATTEMPT TO PUT SAMPLE",sample_queues[0].qsize(), "WORKER", worker_id)
        #print("Worker puts sample",sample_id)
        # Write to leader gpu
        # log.log("Slicer puts sample {}".format(sample_queue.qsize()))
        #print("Attemtpting to put to Sample",sample_queue.qsize())
        #if(sample_queue.qsize ()> num_workers * num_workers * .4):
        #    time.sleep(.1)
        print("Putting into queue", sample_queue.qsize())
        sample_queue.put(tuple(gpu_local_samples))
        # for qid,q in enumerate(sample_queues):
        #     while True:
        #         try:
        #             sample_queues[qid].put_nowait(gpu_local_samples[qid])
        #             print("Worker puts sample", sample_id,"in queue",qid)
        #             break
        #         except:
        #             time.sleep(.0001)
        #             #print("Sampler is too fast, Exception putting stuff")
        # lock.release()

    print("Waiting for sampler process to return")
    while(True):
        time.sleep(1)
        if(sample_queue.qsize()==0):
            break
    time.sleep(30)
    print("SAMPLER PROCESS RETURNS")
