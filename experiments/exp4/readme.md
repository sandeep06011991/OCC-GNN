Summary of metrics I am measuring.
fp.write("graph|hops|naive-partition|cross-edge-comm|cross-node|pa-cache|my-cache|red|skew\n")

pagraph/pagraph_t , me_comm/pagraph_t , \
       me_comm1/pagraph_t, \
           pa_cache_saving/pagraph_t, my_cache_saving/pagraph_t,\
               pa_red/(2 * pa_t), skew

Simulation of possibile benefits of breaking the computation graph.

1. naive-partition.
Each gpu gets a partition of the training vertices,
Assuming that each gpu cache its own partition,
we need to move vertices not in its own partition here.
(Vertices not in its partition)/(Total vertices)

2. Cross edges
Total cross edges vs total fan out in last layer

3. Cross edges communication minimized as nodes.

4. Savings from pagraph.

5. Saving from me

6. Redundant computation nodes as a percentage of all nodes

7. Possible skew in current architecture
max-min/max
