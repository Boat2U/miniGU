use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use dashmap::DashMap;
use diskann::common::AlignedBoxWithSlice;
use diskann::index::{ANNInmemIndex, create_inmem_index};
use diskann::model::IndexConfiguration;
use diskann::model::vertex::{DIM_104, DIM_128, DIM_256};
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use vector::distance_l2_vector_f32;

use super::filter::FilterMask;
use super::index::VectorIndex;
use crate::error::{StorageError, StorageResult, VectorIndexError};

/// Aligned query buffer that maintains 64-byte alignment guarantee
enum AlignedQueryBuffer<'a> {
    Borrowed(&'a [f32]),
    Owned(AlignedBoxWithSlice<f32>),
}

impl AlignedQueryBuffer<'_> {
    fn as_slice(&self) -> &[f32] {
        match self {
            Self::Borrowed(slice) => slice,
            Self::Owned(aligned) => aligned.as_slice(),
        }
    }
}

/// Index statistics and performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IndexStats {
    pub vector_count: usize,
    pub memory_usage: usize,
    pub build_time_ms: Option<u64>,
    pub avg_search_time_us: Option<f64>,
    pub search_count: u64,
    pub brute_force_searches: u64,
    pub post_filter_searches: u64,
    pub pre_filter_searches: u64,
    pub total_brute_force_candidates: u64,
    pub expansion_factor_sum: f64,
    pub expansion_factor_count: u64,
    pub avg_expansion_factor: f64,
    pub max_expansion_factor: usize,
    pub min_expansion_factor: usize,
}

impl IndexStats {
    pub fn new() -> Self {
        Self {
            min_expansion_factor: usize::MAX, // Initialize to max for proper min tracking
            ..Default::default()
        }
    }

    pub fn update_expansion_factor(&mut self, factor: usize) {
        self.expansion_factor_sum += factor as f64;
        self.expansion_factor_count += 1;
        self.avg_expansion_factor = self.expansion_factor_sum / self.expansion_factor_count as f64;
        self.max_expansion_factor = self.max_expansion_factor.max(factor);
        if self.min_expansion_factor == usize::MAX {
            self.min_expansion_factor = factor;
        } else {
            self.min_expansion_factor = self.min_expansion_factor.min(factor);
        }
    }

    pub fn update_after_build(
        &mut self,
        vector_count: usize,
        build_time_ms: u64,
        memory_usage: usize,
    ) {
        self.vector_count = vector_count;
        self.build_time_ms = Some(build_time_ms);
        self.memory_usage = memory_usage;
    }
}

#[allow(clippy::upper_case_acronyms)]
pub struct InMemDiskANNAdapter {
    inner: Box<dyn ANNInmemIndex<f32> + 'static>,
    dimension: usize,

    node_to_vector: DashMap<u64, u32>,
    vector_to_node: DashMap<u32, u64>,
    next_vector_id: AtomicU32, // Next vector ID to be allocated
    stats: std::sync::RwLock<IndexStats>,
}

impl InMemDiskANNAdapter {
    pub fn new(config: IndexConfiguration) -> StorageResult<Self> {
        let dimension = config.dim;
        let inner = create_inmem_index::<f32>(config)
            .map_err(|e| StorageError::VectorIndex(VectorIndexError::DiskANN(e)))?;

        Ok(Self {
            inner,
            dimension, // raw dimension not aligned
            node_to_vector: DashMap::new(),
            vector_to_node: DashMap::new(),
            next_vector_id: AtomicU32::new(0),
            stats: std::sync::RwLock::new(IndexStats::new()),
        })
    }

    pub fn stats(&self) -> IndexStats {
        self.stats
            .read()
            .expect("RwLock poisoned while reading index stats")
            .clone()
    }

    pub fn mapping_count(&self) -> usize {
        self.node_to_vector.len()
    }

    // Private implementation methods for InMemDiskANNAdapter
    fn clear_mappings(&mut self) {
        self.node_to_vector.clear();
        self.vector_to_node.clear();
        self.next_vector_id.store(0, Ordering::Relaxed);
        *self.stats.write().expect(
            "Failed to acquire write lock on stats in clear_mappings (lock may be poisoned)",
        ) = IndexStats::new();
    }

    /// Create aligned query vector for optimal SIMD performance
    /// Uses DiskANN-rs AlignedBoxWithSlice for memory-safe 64-byte alignment
    /// Returns AlignedQueryBuffer to maintain alignment guarantee
    fn ensure_query_aligned(query: &[f32]) -> StorageResult<AlignedQueryBuffer<'_>> {
        if query.as_ptr().align_offset(64) == 0 {
            Ok(AlignedQueryBuffer::Borrowed(query))
        } else {
            let mut aligned = AlignedBoxWithSlice::<f32>::new(query.len(), 64)
                .map_err(|e| StorageError::VectorIndex(VectorIndexError::DiskANN(e)))?;
            aligned.as_mut_slice().copy_from_slice(query);
            Ok(AlignedQueryBuffer::Owned(aligned))
        }
    }

    /// Brute force search with SIMD-optimized distance computation
    /// Direct iteration over candidate vectors for optimal low selectivity performance
    fn brute_force_search(
        &self,
        query: &[f32],
        k: usize,
        filter_mask: &dyn FilterMask,
    ) -> StorageResult<Vec<u64>> {
        if k == 0 {
            return Ok(Vec::new());
        }

        // Ensure query vector is 64-byte aligned for SIMD requirements
        let aligned_query = Self::ensure_query_aligned(query)?;

        let mut heap = BinaryHeap::<(OrderedFloat<f32>, u32)>::with_capacity(k);
        let mut valid_candidates = 0;

        for vector_id in filter_mask.iter_candidates() {
            // TODO: Filter out deleted vectors

            // Get 64-byte aligned vector data from DiskANN (zero-copy access)
            let stored_vector = self
                .inner
                .get_aligned_vector_data(vector_id)
                .map_err(|e| StorageError::VectorIndex(VectorIndexError::DiskANN(e)))?;
            let distance = Self::compute_l2_distance(aligned_query.as_slice(), stored_vector)?;
            valid_candidates += 1;

            if heap.len() < k {
                heap.push((OrderedFloat(distance), vector_id));
            } else if let Some((max_distance, _)) = heap.peek() {
                if OrderedFloat(distance) < *max_distance {
                    heap.pop();
                    heap.push((OrderedFloat(distance), vector_id));
                }
            }
        }
        let results: Vec<_> = heap.into_sorted_vec();

        let node_ids: Vec<u64> = results
            .into_iter()
            .filter_map(|(_, vector_id)| {
                self.vector_to_node.get(&vector_id).map(|node_id| *node_id)
            })
            .collect();

        if let Ok(mut stats) = self.stats.write() {
            stats.brute_force_searches += 1;
            stats.total_brute_force_candidates += valid_candidates;
        }

        Ok(node_ids)
    }

    /// Post-filter search: DiskANN search followed by FilterMask filtering
    /// Used for larger candidate sets where diskann index search is more efficient
    fn post_filter_search(
        &self,
        query: &[f32],
        k: usize,
        l_value: u32,
        filter_mask: &dyn FilterMask,
    ) -> StorageResult<Vec<u64>> {
        let total_nodes = self.size();
        let selectivity = filter_mask.selectivity();

        // Adaptive expansion: use logarithmic scaling for smooth expansion
        let expansion_factor = {
            let log_factor = 2.0 * (-selectivity.ln()).max(1.0);
            (log_factor.ceil() as usize).clamp(2, 50)
        };
        let expanded_k = std::cmp::min(k * expansion_factor, total_nodes);
        if expanded_k == 0 {
            return Ok(Vec::new());
        }

        // Perform regular DiskANN search with expanded k
        let all_results = self.ann_search(query, expanded_k, l_value)?;

        // Filter results using the filter mask's contains_vector method
        // Convert node_id to vector_id for filtering
        let filtered: Vec<u64> = all_results
            .into_iter()
            .filter(|&node_id| {
                if let Some(vector_id) = self.node_to_vector.get(&node_id) {
                    filter_mask.contains_vector(*vector_id)
                } else {
                    false
                }
            })
            .take(k)
            .collect();

        // Update statistics
        if let Ok(mut stats) = self.stats.write() {
            stats.post_filter_searches += 1;
            stats.update_expansion_factor(expansion_factor);
        }

        Ok(filtered)
    }

    /// Compute L2 squared distance between query vector and stored vector
    /// Returns squared distance (without sqrt) for consistency with DiskANN SIMD implementation
    #[inline]
    fn compute_l2_distance(query: &[f32], stored: &[f32]) -> StorageResult<f32> {
        if query.len() != stored.len() {
            return Err(StorageError::VectorIndex(
                VectorIndexError::InvalidDimension {
                    expected: stored.len(),
                    actual: query.len(),
                },
            ));
        }

        let dimension = query.len();

        // Helper macro to safely compute SIMD distance for supported dimensions
        macro_rules! simd_distance {
            ($const_dim:expr) => {{
                // Check 64-byte alignment (Vector crate requirement)
                if query.as_ptr().align_offset(64) != 0 {
                    panic!("query must be 64-byte aligned");
                    // return Ok(Self::compute_scalar_l2_squared(query, stored));
                }
                if stored.as_ptr().align_offset(64) != 0 {
                    panic!("vectors must be 64-byte aligned");
                }

                // Safety: We've verified dimension match and 64-byte alignment
                unsafe {
                    let query_array = &*(query.as_ptr() as *const [f32; $const_dim]);
                    let stored_array = &*(stored.as_ptr() as *const [f32; $const_dim]);
                    distance_l2_vector_f32::<$const_dim>(query_array, stored_array)
                }
            }};
        }

        let distance = match dimension {
            DIM_104 => simd_distance!(DIM_104),
            DIM_128 => simd_distance!(DIM_128),
            DIM_256 => simd_distance!(DIM_256),
            _ => unreachable!(),
        };

        Ok(distance)
    }
}

impl VectorIndex for InMemDiskANNAdapter {
    fn build(&mut self, vectors: &[(u64, Vec<f32>)]) -> StorageResult<()> {
        let start = Instant::now();

        if vectors.is_empty() {
            return Err(StorageError::VectorIndex(VectorIndexError::EmptyDataset));
        }

        self.clear_mappings();

        let mut sorted_vectors = vectors.to_vec();
        sorted_vectors.sort_by_key(|(node_id, _)| *node_id);

        // Validate node IDs and establish ID mappings BEFORE calling DiskANN
        let mut vector_data = Vec::with_capacity(sorted_vectors.len());
        let mut seen_nodes = std::collections::HashSet::new();

        for (array_index, (node_id, vector)) in sorted_vectors.iter().enumerate() {
            // Check for VertexId overflow (DiskANN requires u32 vector IDs)
            if *node_id > u32::MAX as u64 {
                self.clear_mappings();
                return Err(StorageError::VectorIndex(
                    VectorIndexError::VertexIdOverflow {
                        vertex_id: *node_id,
                    },
                ));
            }

            // Check for duplicate node IDs
            if !seen_nodes.insert(*node_id) {
                self.clear_mappings();
                return Err(StorageError::VectorIndex(
                    VectorIndexError::DuplicateNodeId { node_id: *node_id },
                ));
            }

            // Establish ID mapping - DiskANN will assign vector_id = array_index
            let vector_id = array_index as u32;

            self.node_to_vector.insert(*node_id, vector_id);
            self.vector_to_node.insert(vector_id, *node_id);

            vector_data.push(vector.as_slice());
        }

        // Call DiskANN to build the index
        match self.inner.build_from_memory(&vector_data) {
            Ok(()) => {
                self.next_vector_id
                    .store(sorted_vectors.len() as u32, Ordering::Relaxed);

                let build_time = start.elapsed().as_millis() as u64;
                {
                    let mut stats = self.stats.write().expect("Failed to acquire write lock on stats (lock poisoned) while updating build stats");
                    stats.update_after_build(sorted_vectors.len(), build_time, 0);
                }

                Ok(())
            }
            Err(e) => {
                self.clear_mappings();
                Err(StorageError::VectorIndex(VectorIndexError::BuildError(
                    e.to_string(),
                )))
            }
        }
    }

    fn ann_search(&self, query: &[f32], k: usize, l_value: u32) -> StorageResult<Vec<u64>> {
        // Check if index is built
        if self.vector_to_node.is_empty() {
            return Err(StorageError::VectorIndex(VectorIndexError::IndexNotBuilt));
        }

        // Perform DiskANN search
        let effective_k = std::cmp::min(k, self.size());
        if effective_k == 0 {
            return Ok(Vec::new()); // No active vectors
        }

        let mut vector_ids = vec![0u32; effective_k];
        let actual_count = self
            .inner
            .search(query, effective_k, l_value, &mut vector_ids)
            .map_err(|e| StorageError::VectorIndex(VectorIndexError::SearchError(e.to_string())))?;

        // Filter deleted vectors and convert to node_ids
        let mut node_ids = Vec::with_capacity(actual_count as usize);

        for &vector_id in vector_ids.iter().take(actual_count as usize) {
            if let Some(entry) = self.vector_to_node.get(&vector_id) {
                let node_id = *entry;
                // DiskANN-rs already filters deleted vectors in its search method
                // No need for additional filtering here
                node_ids.push(node_id);
            } else {
                // This should not happen if our mapping is consistent
                return Err(StorageError::VectorIndex(
                    VectorIndexError::VectorIdNotFound { vector_id },
                ));
            }
        }

        {
            let mut stats = self.stats.write().expect(
                "Failed to acquire write lock on stats (lock poisoned) while updating search stats",
            );
            stats.search_count += 1;
        }

        Ok(node_ids)
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        l_value: u32,
        filter_mask: Option<&dyn FilterMask>,
    ) -> StorageResult<Vec<u64>> {
        // No filter provided, use regular DiskANN search
        let Some(mask) = filter_mask else {
            return self.ann_search(query, k, l_value);
        };

        // Check if index is built
        if self.vector_to_node.is_empty() {
            return Err(StorageError::VectorIndex(VectorIndexError::IndexNotBuilt));
        }

        // If no valid candidates, return empty
        if mask.candidate_count() == 0 {
            return Ok(Vec::new());
        }

        let selectivity = mask.selectivity();

        // Adaptive strategy selection based on selectivity
        if selectivity < 0.1 {
            self.brute_force_search(query, k, mask)
        } else {
            // For larger candidate sets, use DiskANN with post-filtering
            self.post_filter_search(query, k, l_value, mask)
        }
    }

    fn get_dimension(&self) -> usize {
        self.dimension
    }

    fn size(&self) -> usize {
        // Return the actual number of active vectors based on our mappings
        // This correctly excludes deleted vectors, unlike get_num_active_pts()
        self.node_to_vector.len()
    }

    fn node_to_vector_id(&self, node_id: u64) -> Option<u32> {
        self.node_to_vector.get(&node_id).map(|entry| *entry)
    }

    fn insert(&mut self, vectors: &[(u64, Vec<f32>)]) -> StorageResult<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        if self.node_to_vector.is_empty() {
            return Err(StorageError::VectorIndex(VectorIndexError::IndexNotBuilt));
        }

        // Check for overflow and duplicate node IDs
        for (node_id, _) in vectors {
            // Check for VertexId overflow (DiskANN requires u32 vector IDs)
            if *node_id > u32::MAX as u64 {
                return Err(StorageError::VectorIndex(
                    VectorIndexError::VertexIdOverflow {
                        vertex_id: *node_id,
                    },
                ));
            }

            if self.node_to_vector.contains_key(node_id) {
                return Err(StorageError::VectorIndex(
                    VectorIndexError::DuplicateNodeId { node_id: *node_id },
                ));
            }
        }

        // atomic ID allocation
        let base_vector_id = self
            .next_vector_id
            .fetch_add(vectors.len() as u32, Ordering::Relaxed);

        let mut inserted_mappings = Vec::new();
        for (array_index, (node_id, _)) in vectors.iter().enumerate() {
            let vector_id = base_vector_id + array_index as u32;

            self.node_to_vector.insert(*node_id, vector_id);
            self.vector_to_node.insert(vector_id, *node_id);

            // Track for potential rollback
            inserted_mappings.push((*node_id, vector_id));
        }

        let vector_data: Vec<&[f32]> = vectors
            .iter()
            .map(|(_, vector)| vector.as_slice())
            .collect();

        // Call DiskANN insert
        match self.inner.insert_from_memory(&vector_data) {
            Ok(()) => Ok(()),
            Err(e) => {
                for (node_id, vector_id) in inserted_mappings {
                    self.node_to_vector.remove(&node_id);
                    self.vector_to_node.remove(&vector_id);
                }

                self.next_vector_id
                    .fetch_sub(vectors.len() as u32, Ordering::Relaxed);

                Err(StorageError::VectorIndex(VectorIndexError::BuildError(
                    e.to_string(),
                )))
            }
        }
    }

    fn soft_delete(&mut self, node_ids: &[u64]) -> StorageResult<()> {
        if node_ids.is_empty() {
            return Ok(());
        }

        if self.node_to_vector.is_empty() {
            return Err(StorageError::VectorIndex(VectorIndexError::IndexNotBuilt));
        }

        // Validate all node_ids exist and collect vector_ids to delete
        let mut vector_ids_to_delete = Vec::with_capacity(node_ids.len());
        for &node_id in node_ids {
            if let Some(vector_id) = self.node_to_vector.get(&node_id) {
                // Check if mapping exists in vector_to_node (should always exist if node_to_vector
                // exists)
                if self.vector_to_node.contains_key(&*vector_id) {
                    vector_ids_to_delete.push(*vector_id);
                } else {
                    return Err(StorageError::VectorIndex(
                        VectorIndexError::NodeIdNotFound { node_id },
                    ));
                }
            } else {
                return Err(StorageError::VectorIndex(
                    VectorIndexError::NodeIdNotFound { node_id },
                ));
            }
        }

        // Call DiskANN soft deletion
        match self
            .inner
            .soft_delete(vector_ids_to_delete.clone(), vector_ids_to_delete.len())
        {
            Ok(()) => {
                // DiskANN soft deletion successful, now clean up our mappings
                for &node_id in node_ids {
                    if let Some((_, vector_id)) = self.node_to_vector.remove(&node_id) {
                        // Remove both directions of the mapping
                        self.vector_to_node.remove(&vector_id);
                    }
                }
            }
            Err(e) => {
                // DiskANN soft deletion failed, don't modify our mappings
                return Err(StorageError::VectorIndex(VectorIndexError::DiskANN(e)));
            }
        }

        Ok(())
    }

    fn save(&mut self, _path: &str) -> StorageResult<()> {
        Err(StorageError::VectorIndex(VectorIndexError::NotSupported(
            "save() is not yet implemented".to_string(),
        )))
    }

    fn load(&mut self, _path: &str) -> StorageResult<()> {
        Err(StorageError::VectorIndex(VectorIndexError::NotSupported(
            "load() is not yet implemented for InMemDiskANNAdapter".to_string(),
        )))
    }
}
