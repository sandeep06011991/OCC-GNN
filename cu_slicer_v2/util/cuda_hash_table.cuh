#pragma once
#include "../util/types.h"
#include <torch/torch.h>

// Copied and modified from DGL source code 
// The main idea is retained while modifying it
// while array is index -> value . 
// This table allows us to do value -> index

class DeviceOrderedHashTable {
 
public:
    struct Mapping {
        NDTYPE key;
        NDTYPE local;
        int64_t index;  
    };

    typedef const Mapping * ConstIterator;
    typedef Mapping * Iterator; 

    DeviceOrderedHashTable(){}

    explicit DeviceOrderedHashTable(const Mapping * mapping, size_t size){
        this->table_ = mapping;
        this->size_ = size;
    }

    inline __device__ ConstIterator Search(const NDTYPE id) const {
        const NDTYPE pos = SearchForPosition(id);
        return &table_[pos];
    }

    inline __device__ bool Contains(const NDTYPE id) const {
        NDTYPE pos = Hash(id);
        NDTYPE delta = 1;
        while (table_[pos].key != kEmptyKey) {
            if (table_[pos].key == id) {
            return true;
            }
            pos = Hash(pos + delta);
            delta += 1;
        }
        return false;
    }

    inline __device__ NDTYPE SearchForPosition(const NDTYPE id) const {
        NDTYPE pos = Hash(id);
        // linearly scan for matching entry
        NDTYPE delta = 1;
        while (table_[pos].key != id) {
            assert(table_[pos].key != kEmptyKey);
            pos = Hash(pos + delta);
            delta += 1;
        }
        assert(pos < size_);
        return pos;
  }
    
     inline __device__ bool AttemptInsertAt(
      const size_t pos, const NDTYPE id, const size_t index) {
        const NDTYPE key = atomicCAS(&GetMutable(pos)->key, kEmptyKey, id);
    if (key == kEmptyKey || key == id) {
    
      // we either set a match key, or found a matching key, so then place the
      // minimum index in position. Match the type of atomicMin, so ignore
      // linting
      atomicMin(
          reinterpret_cast<unsigned long long*>(    // NOLINT
              &GetMutable(pos)->index),
          static_cast<unsigned long long>(index));  // NOLINT
      return true;
    } else {
      // we need to search elsewhere
      return false;
    }
  }

     inline __device__ Iterator Insert(const NDTYPE id, const size_t index) {
    size_t pos = Hash(id);

    // linearly scan for an empty slot or matching entry
    NDTYPE delta = 1;
        
    while (!AttemptInsertAt(pos, id, index)) {
      pos = Hash(pos + delta);
      delta += 1;
    }

    return GetMutable(pos);
  }

    inline __device__ Iterator GetMutable(const size_t pos) {
        assert(pos < this->size_);
        return const_cast<Iterator>(this->table_ + pos);
    }
    
    inline __device__ size_t Hash(const NDTYPE id) const { 
        return id % size_; 
    }
  
    const Mapping* table_;
    // Number of slots 
    size_t size_;

    static constexpr NDTYPE kEmptyKey = static_cast<NDTYPE>(-1);

    friend class OrderedHashTable;
};
 
class OrderedHashTable {
 public:
  static constexpr int kDefaultScale = 3;

  using Mapping = typename DeviceOrderedHashTable::Mapping;
  
  OrderedHashTable(const size_t size, const int scale = kDefaultScale){
    c10::TensorOptions opts;
    std::cout << "Hard coded device \n";
    if(sizeof(NDTYPE) == 4){
        opts = torch::TensorOptions().dtype(torch::kInt32)\
            .device(torch::kCUDA, 0);
            }else{
            assert(false);
        }
        // Todo Is this order most optimal
    const size_t next_pow2 = 1 << static_cast<size_t>(1 + std::log2(size >> 1));
    const size_t table_size = static_cast<size_t>(next_pow2 << scale);
    const size_t new_size = (sizeof(Mapping)/sizeof(NDTYPE))  * table_size;
    // unsigned long t = (unsigned long )new_size;
    std::cout << "allocating size " << new_size <<"\n";
    this->table_storage = torch::empty(\
        { (signed long)new_size,}, opts );
    this->table_storage.fill_(DeviceOrderedHashTable::kEmptyKey);

    this->_table = DeviceOrderedHashTable((Mapping *)(this->table_storage.data_ptr()), table_size);
    std::cout << "table size " << table_size << " " << new_size <<" " <<  sizeof(Mapping)  << " " << this->table_storage.data_ptr() << "\n";
    std::cout << "Table ptr" << this->_table.table_ <<"\n";
    std::cout << "After cast " << (Mapping *)this->table_storage.data_ptr() <<"\n";
    this->size_ = table_size;
  }

  ~OrderedHashTable(){}

  // Disable copying
  OrderedHashTable(const OrderedHashTable& other) = delete;
  OrderedHashTable& operator=(const OrderedHashTable& other) = delete;
  
  // and returns unique 
//   returns tensor 
  void FillWithDuplicates(
      const NDTYPE* const input, const size_t num_input,\
         NDTYPE* const unique, int64_t* const num_unique);

  void Replace(NDTYPE * input, const size_t num_input);
//   void FillWithUnique(
//       const NDTYPE* const input, const size_t num_input);


 private:
  torch::Tensor table_storage;
  DeviceOrderedHashTable _table;
  size_t size_;

};
