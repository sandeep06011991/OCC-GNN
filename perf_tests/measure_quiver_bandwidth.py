def test_feature_basic():
    rank = 0

    NUM_ELEMENT = 1000000
    SAMPLE_SIZE = 80000
    FEATURE_DIM = 600

    #########################
    # Init With Numpy
    ########################
    torch.cuda.set_device(rank)

    host_tensor = np.random.randint(0,
                                    high=10,
                                    size=(2 * NUM_ELEMENT, FEATURE_DIM))
    tensor = torch.from_numpy(host_tensor).type(torch.float32)
    host_indice = np.random.randint(NUM_ELEMENT, 2 * NUM_ELEMENT - 1, (SAMPLE_SIZE, ))
    indices = torch.from_numpy(host_indice).type(torch.long)
    print("host data size", host_tensor.size * 4 // 1024 // 1024, "MB")

    device_indices = indices.to(rank)

    ############################
    # define a quiver.Feature
    ###########################
    feature = quiver.Feature(rank=rank,
                             device_list=[0, 1, 2, 3],
                             device_cache_size="0.9G",
                             cache_policy="p2p_clique_replicate")
    feature.from_cpu_tensor(tensor)
    offsets = feature.clique_tensor_list[0].shard_tensor_config.tensor_offset_device[3]
    start = offsets.start
    end = offsets.end
    print(start,end)
    device_indices = torch.randint(start, end, (SAMPLE_SIZE,)).to(rank)
    ####################
    # Indexing
    ####################
    res = feature[device_indices]

    start = time.time()
    e1 = torch.cuda.Event(enable_timing = True)
    e2 = torch.cuda.Event(enable_timing = True)
    start = time.time()
    e1.record()
    res = feature[device_indices]
    e2.record()
    e2.synchronize()
    print("Event time",e1.elapsed_time(e2)/1000) 
    consumed_time = time.time() - start
    res = res.cpu().numpy()
    feature_gt = tensor[indices].numpy()
    print("Correctness Check : ", np.array_equal(res, feature_gt))
    print(
        f"Process {os.getpid()}: TEST SUCCEED!, With Memory Bandwidth = {res.size * 4 / consumed_time / 1024 / 1024 / 1024} GB/s, consumed {consumed_time}s"
    )


