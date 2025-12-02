# HNSW Algorithm Analysis

## Overview

This document analyzes the HNSW (Hierarchical Navigable Small World) algorithm implementation in ParlayANN, with a focus on:
1. Visited set implementation
2. Locking and synchronization mechanisms
3. Read/write contention handling
4. Cross-reference with the ParlayANN paper (https://arxiv.org/abs/2305.04359)

## Algorithm Structure

The HNSW implementation in this repository (`algorithms/HNSW/HNSW.hpp`) uses a hierarchical graph-based approach for approximate nearest neighbor search. The key components are:

### Core Data Structures

```cpp
struct node {
    uint32_t level;                          // Layer level in the hierarchy
    parlay::sequence<node_id> *neighbors;    // Adjacency lists for each layer
    T data;                                  // The actual point data
};

parlay::sequence<node> node_pool;            // All nodes in the index
parlay::sequence<node_id> entrance;          // Entry points for search
```

### Key Observations

1. **Lock-Free Design**: The implementation uses a **batch-based, lock-free approach** rather than fine-grained locking
2. **Parallel Construction**: Uses ParlayLib's parallel primitives (`parlay::parallel_for`) for parallel batch processing
3. **No Atomic Operations**: Notably absent are `std::atomic`, `std::mutex`, or explicit locks

## Visited Set Implementation

### 1. Hash-Based Visited Tracking

The visited set implementation varies between the search and construction phases:

#### During Search (`search_layer` function, lines 1089-1109)

The implementation delegates to `beam_search_impl` from `beamSearch.h`, which uses:

```cpp
hashset<indexType> has_been_seen(2 * (10 + beamSize) * max_degree);
```

This is a **filtered hashset** (defined in `algorithms/utils/filtered_hashset.h`) with the following characteristics:

- **Lock-free hash table** with linear probing
- **Approximate membership**: Can give false negatives but not false positives
- **Dynamic resizing**: Grows when load factor exceeds 50%
- **Thread-local**: Each search operates on its own hashset instance

```cpp
// From filtered_hashset.h
template <typename intT>
struct hashset {
    int bits;
    std::vector<intT> filter;
    size_t mask;
    long num_entries = 0;
    
    bool operator () (intT a) {
        int loc = hash(a) & mask;
        if (filter[loc] == a) return true;
        if (filter[loc] != -1) {
            loc = (loc + 1) & mask;
            while (filter[loc] != -1 && filter[loc] != a)
                loc = (loc + 1) & mask;
            if (filter[loc] == a) return true;
        }
        // ... insertion logic ...
    }
};
```

**Key Properties**:
- **O(1) average-case** lookup and insertion
- **No synchronization required** (thread-local)
- **Space efficient**: Uses power-of-2 sized array with linear probing

#### Alternative Implementation (search_layer_bak, lines 1112-1249)

The backup implementation shows three different visited tracking strategies:

1. **Hash Table Approach** (`USE_HASHTBL`):
```cpp
const uint32_t bits = ef>2? std::ceil(std::log2(ef*ef))-2: 2;
const uint32_t mask = (1u<<bits)-1;
parlay::sequence<uint32_t> visited(mask+1, n+1);
```
- Uses a compact hash table with sentinel values
- Hash collisions handled by checking if `visited[hash(id)] == id`

2. **Boolean Array** (`USE_BOOLARRAY`):
```cpp
std::vector<bool> visited(n+1);
```
- Simple O(n) space array
- Direct indexing for O(1) access

3. **Unordered Set** (`USE_UNORDERED_SET`):
```cpp
std::unordered_set<uint32_t> visited;
```
- Standard STL set for visited tracking

### 2. Visited Set During Beam Search

The beam search implementation (`algorithms/utils/beamSearch.h`) maintains:

```cpp
std::vector<id_dist> frontier;           // Current best candidates (size ≤ beamSize)
std::vector<id_dist> unvisited_frontier; // Not yet explored
std::vector<id_dist> visited;            // All visited nodes (sorted)
hashset<indexType> has_been_seen;        // Fast membership check
```

**Search Process**:
1. Start with entry points in `frontier`
2. Pick closest unvisited point from `unvisited_frontier`
3. Check neighbors against `has_been_seen` hashset
4. Add new candidates to `frontier` and `visited`
5. Maintain sorted order for efficient set operations

## Locking and Synchronization Mechanisms

### Lock-Free Batch-Based Approach

**Critical Insight**: This implementation **avoids traditional locks entirely** through a batch-based construction algorithm.

### Construction Phase (insert function, lines 828-1014)

The insertion algorithm processes batches of points in **three coordinated phases**:

#### Phase 1: Node Creation (Parallel, Lock-Free)
```cpp
parlay::parallel_for(0, size_batch, [&](uint32_t i){
    const T &q = *(begin+i);
    const auto level_u = get_level_random();
    node_id pu = offset+i;
    
    new(&get_node(pu)) node{
        level_u,
        new parlay::sequence<node_id>[level_u+1],
        q
    };
    node_new[i] = pu;
});
```

**No contention**: Each thread writes to a disjoint memory location (`offset+i`)

#### Phase 2: Entry Point Search (Parallel, Read-Only)
```cpp
parlay::parallel_for(0, size_batch, [&](uint32_t i){
    auto &u = get_node(node_new[i]);
    const auto level_u = u.level;
    auto &eps_u = eps[i]; 
    eps_u = entrance;
    for(uint32_t l=level_ep; l>level_u; --l) {
        const auto res = search_layer(u, eps_u, 1, l);
        eps_u.clear();
        eps_u.push_back(res[0].u);
    }
});
```

**No contention**: Reads from shared graph structure, writes to thread-local `eps[i]`

#### Phase 3: Edge Addition (Layer-by-Layer Synchronization)

This is the **most sophisticated part** where potential write conflicts are resolved:

```cpp
for(int32_t l_c=level_ep; l_c>=0; --l_c) {
    // Step 3a: Find forward edges (Parallel, No Conflicts)
    parlay::parallel_for(0, size_batch, [&](uint32_t i){
        node_id pu = node_new[i];
        auto &u = get_node(pu);
        if((uint32_t)l_c>u.level) return;
        
        auto res = search_layer(u, eps_u, ef_construction, l_c);
        auto neighbors_vec = select_neighbors(u.data, res, get_threshold_m(l_c), l_c);
        
        for(node_id pv : neighbors_vec)
            edge_u.emplace_back(pv, pu);
        nbh_new[i] = std::move(neighbors_vec);
    });
    
    // Step 3b: Add forward edges (Parallel, No Conflicts)
    parlay::parallel_for(0, size_batch, [&](uint32_t i){
        auto &u = get_node(node_new[i]);
        if((uint32_t)l_c<=u.level)
            neighbourhood(u,l_c) = std::move(nbh_new[i]);
    });
    
    // Step 3c: Group and add reverse edges (Parallel with Coordination)
    auto edge_add_flatten = parlay::flatten(edge_add);
    auto edge_add_grouped = parlay::group_by_key(edge_add_flatten);
    
    parlay::parallel_for(0, edge_add_grouped.size(), [&](size_t j){
        node_id pv = edge_add_grouped[j].first;
        auto &nbh_v = neighbourhood(get_node(pv),l_c);
        auto &nbh_v_add = edge_add_grouped[j].second;
        
        const uint32_t size_nbh_total = nbh_v.size()+nbh_v_add.size();
        const auto m_s = get_threshold_m(l_c)*factor_m;
        
        if(size_nbh_total>m_s) {
            // Prune edges using distance-based heuristic
            auto candidates = parlay::sequence<dist>(size_nbh_total);
            // ... compute distances ...
            std::sort(candidates.begin(), candidates.end(), farthest());
            nbh_v.resize(m_s);
            for(size_t k=0; k<m_s; ++k)
                nbh_v[k] = candidates[k].u;
        }
        else {
            nbh_v.insert(nbh_v.end(), nbh_v_add.begin(), nbh_v_add.end());
        }
    });
}
```

### Key Synchronization Strategy: `group_by_key`

The **critical synchronization primitive** is:
```cpp
auto edge_add_grouped = parlay::group_by_key(edge_add_flatten);
```

This ParlayLib function:
1. **Flattens** all edge additions from the batch: `[(v1→u1), (v2→u1), (v1→u2), ...]`
2. **Groups** by target vertex: `{v1: [u1, u2], v2: [u1], ...}`
3. **Eliminates conflicts**: Each resulting group is processed by a single thread

**Result**: No two threads write to the same vertex's neighbor list simultaneously.

## Contention Points and Resolution

### 1. Neighbor List Updates (PRIMARY CONTENTION POINT)

**Potential Conflict**: Multiple new nodes may want to add edges to the same existing node.

**Resolution**: 
- **Batch and group**: Collect all edges to same target, process once
- **Serial pruning per vertex**: Each vertex's neighbor list updated by single thread
- **No locks needed**: Conflict-free through data partitioning

### 2. Entry Point Updates (MINOR CONTENTION POINT)

**Location**: Lines 991-1007
```cpp
node_id node_highest = *std::max_element(
    node_new.get(), node_new.get()+size_batch, [&](const node_id u, const node_id v){
        return get_node(u).level < get_node(v).level;
});
if(get_node(node_highest).level>level_ep) {
    entrance.clear();
    entrance.push_back(node_highest);
}
```

**Resolution**: 
- **Serial update**: Happens after all parallel work completes
- **No synchronization needed**: Single-threaded at this point

### 3. Node Pool Allocation

**Potential Conflict**: Growing `node_pool` during insertion.

**Resolution**:
- **Pre-allocation**: `node_pool.resize(offset+size_batch)` before parallel work
- **Disjoint writes**: Each thread writes to `node_pool[offset+i]`

## Read/Write Contention Analysis

### Read Operations (Query Phase)

**Concurrent reads are safe** because:
1. **Immutable during queries**: Graph structure doesn't change
2. **No memory barriers needed**: Reads from stable memory locations
3. **Thread-local visited sets**: No sharing between queries

### Write Operations (Construction Phase)

**Write conflicts avoided through**:

#### 1. **Temporal Partitioning**
```
Time 1: Create new nodes (disjoint writes to node_pool)
Time 2: Search for neighbors (read-only on existing graph)
Time 3: Update forward edges (writes to new nodes only)
Time 4: Update reverse edges (partitioned by target vertex)
```

#### 2. **Spatial Partitioning**
- New nodes: Each thread owns disjoint subset
- Existing nodes: `group_by_key` ensures exclusive access per vertex

#### 3. **Layer-by-Layer Processing**
```cpp
for(int32_t l_c=level_ep; l_c>=0; --l_c) {
    // Process entire batch at layer l_c
    // Then barrier (implicit in parlay::parallel_for)
    // Then move to layer l_c-1
}
```

This ensures **all threads complete layer L before any start layer L-1**.

### Mixed Read/Write Scenarios

**Not explicitly handled** in this implementation:
- The code assumes **batch construction** followed by **query phase**
- No support for **concurrent inserts and queries**
- This is a **common design pattern** in ANNS indexes to avoid complex synchronization

## Comparison with HNSW Paper and ParlayANN Paper

### Standard HNSW (Malkov & Yashunin 2016)

The original HNSW uses:
- **Fine-grained locks** on each vertex's neighbor list
- **Concurrent insertions** with point-wise locking
- **Read-write locks** to allow concurrent queries during construction

### ParlayANN's Lock-Free HNSW

This implementation makes a **different trade-off**:

**Advantages**:
1. **No lock overhead**: Eliminates lock acquisition/release costs
2. **Better cache behavior**: Batch processing improves locality
3. **Scalability**: No lock contention as thread count increases
4. **Deterministic**: Same input always produces same graph

**Trade-offs**:
1. **Batch-only construction**: Cannot insert single points efficiently
2. **No concurrent insert/query**: Must complete batch before querying
3. **Memory overhead**: Temporary arrays for batch processing

### Connection to ParlayANN Paper (arXiv:2305.04359)

The paper "Scaling Graph-Based ANNS Algorithms to Billion-Size Datasets" describes this approach:

**Key Innovation**: "We design parallel, lock-free versions of these algorithms"

**Strategy** (from our code analysis):
1. **Batch parallelism**: Process multiple insertions together
2. **Phase separation**: Divide insertion into conflict-free phases
3. **Grouping primitives**: Use `group_by_key` for conflict resolution
4. **No locks**: Achieve thread-safety through algorithmic design

**Quote from README**:
> "This repository was built for our paper Scaling Graph-Based ANNS Algorithms to Billion-Size Datasets: A Comparative Analysis"

The HNSW implementation exemplifies the paper's approach of using:
- **Parallel primitives** instead of locks
- **Batch processing** for amortized efficiency
- **Deterministic algorithms** for reproducibility

## Performance Implications

### Visited Set Performance

**Memory Footprint**:
- Hashset: O(beamSize × max_degree) per query
- Thread-local: No sharing overhead
- Dynamic resizing: Adapts to query difficulty

**Lookup Performance**:
- Average: O(1) with low constant
- Worst case: O(max_degree) with linear probing
- Cache-friendly: Power-of-2 sizing helps

### Synchronization Performance

**Batch Construction**:
- **Pros**: 
  - No lock contention
  - Good cache locality from batch processing
  - Scales linearly with thread count
- **Cons**:
  - Higher latency for small batches
  - Cannot overlap construction with queries

**Query Performance**:
- **Lock-free reads**: No synchronization overhead
- **Thread-local state**: No cache-line ping-pong
- **Parallel queries**: Linear scaling with thread count

## Summary of Locking Strategy

### What Requires Synchronization?

1. **Neighbor list updates**: YES (via batch grouping)
2. **Node pool growth**: YES (via pre-allocation)
3. **Entry point updates**: YES (via serial execution)
4. **Visited set access**: NO (thread-local)
5. **Graph traversal**: NO (read-only)

### How is Synchronization Achieved?

| Operation | Mechanism | Details |
|-----------|-----------|---------|
| Neighbor updates | `group_by_key` partitioning | Each vertex owned by one thread |
| Node allocation | Pre-sized buffer + disjoint indices | No overlap in write locations |
| Entry point update | Sequential execution | After parallel phase completes |
| Layer transitions | Implicit barriers | `parallel_for` synchronizes at end |

### No Traditional Locks Because:

1. **Batch processing**: Collect all conflicts, resolve once
2. **Algorithmic partitioning**: Design ensures no concurrent writes to same location
3. **ParlayLib primitives**: Built-in synchronization via parallel patterns
4. **Phase-based execution**: Separate conflicting operations temporally

## Conclusion

The ParlayANN HNSW implementation demonstrates a **lock-free, batch-oriented approach** to parallel ANNS construction:

1. **Visited sets** use lock-free hash tables with thread-local instances
2. **No explicit locks** (mutex, atomic) in the codebase
3. **Synchronization via batching**: `group_by_key` eliminates conflicts
4. **Read/write separation**: Construction and query phases don't overlap
5. **Scalable design**: Linear speedup with thread count for both construction and queries

This design achieves **high performance** through algorithmic techniques rather than fine-grained synchronization, at the cost of flexibility in supporting concurrent insertions during queries.
