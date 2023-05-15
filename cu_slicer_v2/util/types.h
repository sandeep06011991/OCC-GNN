#pragma once


#define  INTMAX 2147483647


// Ogbn-products sample charectersitcs
// Nodes [1705194, 684535, 76689, 4096] Edges [12975036, 1490753, 80134]
// Flattenned array takes .5 gb for all partitions
// Reorder - map 
// Nodes [753565, 226407, 36803, 4096] Edges [1743108, 308245, 33447]
// typedef short PARTITIONIDX ;
typedef int PARTITIONIDX;
typedef int NDTYPE;
// typedef  int NDTYPE;
// All graphs have less than 2 billion edges