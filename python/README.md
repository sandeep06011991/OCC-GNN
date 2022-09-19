Mistake.
Multiple process synchronization is hard without python primitaves.
I need one layer of python on top for this.




1. no_cache_multi_gpu.py
    -- Naive version of no-cache multi-gpu
2. pa_cache_multi_gpu.py
    -- Version with caching high degree vertices
    -- Training partition not following pagraph
    -- or high degree not computing from K-hop
3. batch_slice_multi_gpu.py


Final end-to-end architecture:
1. train.py --main-entry-point
2. sampler.py
    Goal: Break down sampler to generate partitioned samples
    check performance with naive
    Enable features such as prefetch and multi work.
    Will otherwise definately be a bottleneck.
3. cache.py
    Handles cache management
    Use pinned memory
    Handle varying levels of memory availability.
4. models in final folder
