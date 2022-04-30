
from cslicer import cslicer
graphname = "ogbn-arxiv"
queue_size = 128
no_worker_threads = 32
number_of_epochs = 3
minibatch_size = 512

csl = cslicer(graphname, queue_size, no_worker_threads, number_of_epochs, minibatch_size)
mb = csl.get_expected_number_of_minibatches()

for i in range(mb * no_epochs):
    b = csl.getSample()
    print("generating sample",i)
    # print(b.layers[0][0].in_nodes)
# b = a.getSample()
# print(b)
