import statistics


SAMPLE_START_TIME = 0
GRAPH_LOAD_START_TIME = 1
DATALOAD_START_TIME = 2
DATALOAD_END_TIME = 3
FORWARD_ELAPSED_EVENT_TIME = 4
DATALOAD_ELAPSED_EVENT_TIME = 6
END_BACKWARD = 5

string_map = {}
string_map[0] = "SAMPLE_START_TIME"
string_map[1] = "GRAPH_LOAD_START_TIME"
string_map[2] = "DATALOAD_START_TIME"
string_map[3] = "DATALOAD_END_TIME"
string_map[4] = "FORWARD_ELAPSED_EVENT_TIME"
string_map[5] = "END_BACKWARD"
string_map[6] = "DATALOAD_ELAPSED_EVENT_TIME"

def get_average_ignoring_first(ls):
    ls = ls[1:]
    mean = statistics.mean(ls)
    variance = statistics.stdev(ls)
    # Variance less than mean 5%
    print(abs(variance/mean))
    if(abs(variance/mean) >= .1):
        print(ls)
    # print("Variance/Mean",(variance/mean))
    return mean

def compute_stats_for_minibatch(eventlist_for_gpus):
    for i in eventlist_for_gpus:
        # 4 + 1 from above
        assert(len(i.keys())==7)
    # mb = {}
    # for i in range(4):
    #     for j in (eventlist_for_gpus[i]):
    #         mb[eventlist_for_gpus[i][j]] = string_map[j]+" " + str(i) + " " + str(eventlist_for_gpus[i][j])
    # for i in sorted(mb.keys()):
    #     print(mb[i])
    # print("End")

    min_graph_start = min([e[GRAPH_LOAD_START_TIME] for e in eventlist_for_gpus])
    max_graph_end = max([e[DATALOAD_START_TIME] for e in eventlist_for_gpus])
    max_load_end = max([e[DATALOAD_END_TIME] for e in eventlist_for_gpus])
    min_load_start = min([e[DATALOAD_START_TIME] for e in eventlist_for_gpus])
    max_end_backward = max([e[END_BACKWARD] for e in eventlist_for_gpus])
    start_bw = [e[DATALOAD_END_TIME]  + e[FORWARD_ELAPSED_EVENT_TIME] for e in eventlist_for_gpus]
    batch_load_time = max_load_end - min_load_start
    batch_forward = max(start_bw) - max_load_end
    batch_backward  = max_end_backward - max(start_bw)
    batch_graph = max_graph_end - min_graph_start
    batch_sample = sum([e[GRAPH_LOAD_START_TIME] - e[SAMPLE_START_TIME]  for e in eventlist_for_gpus])/4
    all_load_time = sum([e[DATALOAD_ELAPSED_TIME] for e in eventlist_for_gpus]/4)
    #all_load_time = max((max_load_end - min_graph_start), load_elapsed_time)
    max_epoch_time = max_end_backward - min_graph_start
    return batch_sample, batch_graph, batch_load_time, batch_forward, batch_backward, all_load_time, max_epoch_time

def compute_metrics(recieved_metrics):
    assert(len(recieved_metrics) == 4)
    num_epochs = len(recieved_metrics[0])
    for i in range(0,4):
        assert(len(recieved_metrics[i]) == num_epochs)
    epoch_batch_sample = []
    epoch_batch_graph = []
    epoch_batch_feat_time = []
    epoch_batch_forward = []
    epoch_batch_backward = []
    epoch_batch_loadtime = []
    epoch_batch_totaltime = []
    for epoch in range(num_epochs):
        num_batches = len(recieved_metrics[0][epoch])
        total_sample = 0
        total_graph = 0
        total_feat_time = 0
        total_forward = 0
        total_backward = 0
        total_load_time = 0
        total_epoch_time = 0
        for i in range(4):
            assert(len(recieved_metrics[i][epoch]) == num_batches)
        for batch in range(num_batches):
            eventlist_for_gpus = []
            for j in range(4):
                eventlist_for_gpus.append(recieved_metrics[j][epoch][batch])
            batch_sample, batch_graph, batch_load_time, batch_forward, \
                batch_backward, all_load_time, max_epoch_time= compute_stats_for_minibatch(eventlist_for_gpus)
            total_sample += batch_sample
            total_graph +=  batch_graph
            total_feat_time += batch_load_time
            total_forward += batch_forward
            total_backward += batch_backward
            total_load_time += all_load_time
            total_epoch_time += max_epoch_time
        epoch_batch_sample.append(total_sample)
        epoch_batch_feat_time.append(total_feat_time)
        epoch_batch_graph.append(total_graph)
        epoch_batch_forward.append(total_forward)
        epoch_batch_backward.append(total_backward)
        epoch_batch_loadtime.append(total_load_time)
        epoch_batch_totaltime.append(total_epoch_time)
    # print("sample", epoch_batch_sample)
    # print("graph", epoch_batch_graph)
    # print("load", epoch_batch_load_time)
    # print("forward", epoch_batch_forward)
    # print("backward", epoch_batch_backward)
    epoch_batch_sample = get_average_ignoring_first(epoch_batch_sample)
    epoch_batch_graph = get_average_ignoring_first(epoch_batch_graph)
    epoch_batch_feat_time = get_average_ignoring_first(epoch_batch_feat_time)
    epoch_batch_forward = get_average_ignoring_first(epoch_batch_forward)
    epoch_batch_backward = get_average_ignoring_first(epoch_batch_backward)
    epoch_batch_loadtime = get_average_ignoring_first(epoch_batch_loadtime)
    epoch_batch_totaltime = get_average_ignoring_first(epoch_batch_totaltime)
    return epoch_batch_sample, epoch_batch_graph, epoch_batch_feat_time, \
            epoch_batch_forward, epoch_batch_backward, epoch_batch_loadtime, epoch_batch_totaltime
