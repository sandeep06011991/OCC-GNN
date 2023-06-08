export PYTHONPATH=`pwd`/cslicer:`pwd`/cu_slicer_v2:`pwd`/python:`pwd`/upgraded_pagraph:`pwd`/3rdparty/torch-quiver/srcs/python/
export MKL_THREADING_LAYER=1
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.9
# CUDA 11.7 + 1
# Freeze to Quiver main branch
# For quiver 
export MAX_JOBS=10