## Three major measurements.

1. Measure memcpy accuractly proof with nsys and time.time
Measure asynchronous kernel run time accurately
measure with nsys.
  Time.time might measure from time to lauch.
  cudaEvents measures tighter bounds.
  Closest results to nsys


3. How to measure child processes.
  nsys profile   --trace-fork-before-exec true   python3 mp_test.py

4. How to script wit nsys systems.

5. Use nvtx annotations.

  

Conclusions and best practices.
1. time.time is accuracte with nsys in the absence of async kernels.
