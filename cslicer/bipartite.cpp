#include "bipartite.h"

void BiPartite::reorder(DuplicateRemover* dr){
  dr->order_and_remove_duplicates(in_nodes);
  dr->replace(indices);
  dr->replace(self_ids_in);
  dr->clear();

  dr->order_and_remove_duplicates(out_nodes);
  dr->replace(owned_out_nodes);
  dr->replace(self_ids_out);
  for(int i=0;i<4;i++){
    dr->replace(from_ids[i]);
    dr->replace(to_ids[i]);
  }
  dr->clear();
}
