import pickle
from utils.timing_analysis import *

collected_metrics = []
for i in range(4):
    with open("metrics{}.pkl".format(i), "rb") as input_file:
        cm = pickle.load(input_file)
        collected_metrics.append(cm)
epoch_batch_sample, epoch_batch_graph, epoch_batch_load_time, epoch_batch_forward, epoch_batch_backward = \
compute_metrics(collected_metrics)
print("sample_time:{}".format(epoch_batch_sample))
print("movement graph:{}".format(epoch_batch_graph))
print("movement feature:{}".format(epoch_batch_load_time))
print("forward time:{}".format(epoch_batch_forward))
print("backward time:{}".format(epoch_batch_backward))

events = {}
for i in range(4):
    for epoch in range(len(collected_metrics[i])):
        t = 0
        for j in range(len(collected_metrics[i][epoch])):
            t += (collected_metrics[i][epoch][j][DATALOAD_END_TIME]) - (collected_metrics[i][epoch][j][DATALOAD_START_TIME])
        print("graph load", t, "epoch",epoch, "gpu",i)
