import numpy as np
import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import time
import math
import argparse
import logging


def run(proc_id, n_gpus, args, devices):
    # Start up distributed training, if enabled.
    print("starting sub process", proc_id)
    dev_id = devices[proc_id]
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        th.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=proc_id)
    out = th.cuda.set_device(dev_id)
    filename = "epochs_{}_hidden_{}".format(args.num_epochs, args.num_hidden)
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    if proc_id == 0:
        # create file handler which logs even debug messages
        fh = logging.FileHandler(filename,mode='w')
        fh.setLevel(logging.INFO)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
    print("Threads reached here")
    logger.info("Mark hello world")
    logger.info("{}".format([1,2,3]))
    th.distributed.barrier()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--num-epochs', type=int, default=10)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--batch-size', type=int, default=1032)
    argparser.add_argument('--lr', type=float, default=0.01)
    argparser.add_argument('--dropout', type=float, default=0)
    args = argparser.parse_args()
    n_gpus =3
    devices = [1,2,3]
    start_time = time.time()
    if n_gpus == 1:
        # assert(false)
        print("Running on single GPUs")
        run(0, n_gpus, args, devices, data)
    else:
        procs = []
        print("Launch multiple gpus")
        for proc_id in range(n_gpus):
            p = mp.Process(target=(run),
                           args=(proc_id, n_gpus, args, devices))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
    end_time = time.time()
    print("Total time across all processes", end_time - start_time)
