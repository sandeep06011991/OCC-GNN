# from layers.test_gatconv import test_base, test_dist_bipartite
from data.test_bipartite import serialization_test,serializtion_test_gpu_local_sample
from layers.test_gatconv import test_dist_bipartite

if __name__=="__main__":
    test_dist_bipartite()
    # serializtion_test_gpu_local_sample()
