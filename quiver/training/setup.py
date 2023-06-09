
from utils.utils import *
import logging 
import os 

def setup_logger(rank,mode, args, train_nid, cache_size):
    if rank == 0:
        print(PATH_DIR)
        os.makedirs('{}/quiver/logs_{}'.format( ROOT_DIR, mode,),exist_ok = True)
        FILENAME= ('{}/quiver/logs_{}/{}_{}_{}_{}_uva{}.txt'.format(ROOT_DIR, mode,  \
            args.graph, args.batch_size, args.model, cache_size,  not args.no_uva))
        fileh = logging.FileHandler(FILENAME, 'w')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fileh.setFormatter(formatter)
        log = logging.getLogger()  # root logger
        log.addHandler(fileh)      # set the new handler
        log.setLevel(logging.INFO)
        logging.info(f'To train number of batches {train_nid.shape[0]/args.batch_size}')
    
def get_dataloader(rank, args, shared_graph, train_nid, sampler, idx_split):
    device = rank
    if args.no_uva:
        # copy only the csc to the GPU
        d_graph = shared_graph.to(device)
        train_dataloader = dgl.dataloading.DataLoader(
            d_graph,
            train_nid,
            sampler,
            device=device,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers= 0,
            persistent_workers=False)
        if rank == 0:
            valid_nid = idx_split["valid"]
            valid_dataloader = dgl.dataloading.DataLoader(
                d_graph,
                valid_nid.to(rank),
                sampler,
                device=device,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers= 0,
                persistent_workers=False)
        else: 
            valid_dataloader = None 
    else:
        if rank == 0:
            logging.info("Use UVA True")
            valid_nid = idx_split["valid"].to(rank)
            valid_dataloader = dgl.dataloading.DataLoader(
                shared_graph,
                valid_nid,
                sampler,
                device=device,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers= 0,
                persistent_workers=False, use_uva = True)
        else: 
            valid_dataloader = None 

        train_dataloader = dgl.dataloading.DataLoader(
            shared_graph,
            train_nid,
            sampler,
            device=device,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers= 0,
            persistent_workers=False, use_uva = True)

    return train_dataloader, valid_dataloader
    

def get_model():
    pass 
