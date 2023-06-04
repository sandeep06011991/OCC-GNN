## Flow of the new pipeline 

1. Read OGB/Mag240M/ Dataset, partition using metis and write all files to disk.
Next steps for preprocessing should not touch these external libraries. 
Accessing external libraries is allowed only by baselines for correctness.
If baselines need to use synthetic graphs, they will follow groots pipeline  
2. Offset Book: For a given cache percentage, the offset from each partition map. 
   store as {1G}_cache_map.txt 
