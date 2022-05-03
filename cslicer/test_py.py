# import cslicer
from cslicer import cslicer
# print("Hello world")
graphname = "com-orkut"
queue_size = 16
no_worker_threads = 32
number_of_epochs = 4
minibatch_size =4096
#
csl = cslicer(graphname, queue_size, no_worker_threads, number_of_epochs, minibatch_size)
# # mb = csl.get_expected_number_of_minibatches()
#const std::string &name, int queue_size, int no_worker_threads \
# , int number_of_epochs, int samples_per_epoch, int minibatch_size
# # for i in range(mb * no_epochs):
# import time
# time.sleep(10)
# a = cslicer.test_pyfront()
# print(a)
# print(cslicer.test_list([1,2,3]))

ns = csl.getNoSamples()
print("Checked number of samples", ns)
for i in range(ns):
    s = csl.getSample()
    print(i)
print("Full iteration complete")
