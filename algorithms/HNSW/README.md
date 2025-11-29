# HNSW Algorithm

Hierarchical Navigable Small World (HNSW) is a graph-based approximate nearest neighbor search algorithm that builds a multi-layer structure for efficient similarity search.

## Implementation

The HNSW algorithm is implemented in `HNSW.hpp` as a header-only template library.

### Key Methods

- **`HNSW::search()`** - Main search interface that returns k-nearest neighbors
- **`HNSW::search_layer()`** - Layer-wise beam search traversal
- **`HNSW::insert()`** - Batch insertion of new vectors into the index

### Build Status

This implementation is a header-only template library. To use it, include `HNSW.hpp` directly in your project along with the required ParlayLib dependencies.

**Note:** The `CMakeLists.txt` currently has a TODO for standalone build support.

## Visited List Implementation

The visited list table, which tracks visited nodes during graph traversal, is implemented in `../utils/filtered_hashset.h`.

### Two Variants

1. **`filtered_hashset<intT>`** - Probabilistic hashset (may have false negatives), faster performance
2. **`hashset<intT>`** - Exact hashset with linear probing, no false negatives

### Usage Example

```cpp
// From beamSearch.h
hashset<indexType> has_been_seen(2 * (10 + beamSize) * max_degree);
```

### Implementation Details

Within `HNSW.hpp`, three different visited list approaches are conditionally compiled:

1. **Hash table** (active) - Uses `parlay::hash64_2()` with masking
2. **Boolean array** - Uses `std::vector<bool>`
3. **Unordered set** - Uses `std::unordered_set<uint32_t>`

The hash table approach is currently the default active implementation for its balance of speed and accuracy.

## Dependencies

- [ParlayLib](https://github.com/cmuparlay/parlaylib) - Parallel algorithms library
