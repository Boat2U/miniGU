use std::sync::Arc;

use arrow::array::{BooleanArray, UInt64Array};
use minigu_catalog::provider::{GraphProvider, GraphTypeProvider};
use minigu_common::data_chunk;
use minigu_common::data_type::{DataField, DataSchema, LogicalType};
use minigu_common::value::ScalarValue;
use minigu_context::graph::{GraphContainer, GraphStorage};
use minigu_context::procedure::Procedure;
use minigu_context::session::SessionContext;
use minigu_storage::tp::{IsolationLevel, MemoryGraph};

/// VectorSearch procedure for performing vector similarity search with optional filtering.
///
/// Function signature: CALL VectorSearch(property_name: String, query_vector: Vec<f32>,
/// k: u32, l_value: u32, filter_condition: String?)
///
/// Returns: A list of node IDs sorted by similarity (most similar first)
pub fn build_procedure() -> Procedure {
    let parameters = vec![
        LogicalType::String,    // property_name
        LogicalType::Vector(0), // query_vector (dimension validated at runtime)
        LogicalType::UInt32,    // k
        LogicalType::UInt32,    // l_value
        LogicalType::String,    // filter_condition (optional)
    ];
    let schema = Arc::new(DataSchema::new(vec![DataField::new(
        "node_id".into(),
        LogicalType::UInt64,
        false,
    )]));

    Procedure::new(parameters, Some(schema), |context, args| {
        assert_eq!(args.len(), 5);

        let property_name = args[0]
            .try_as_string()
            .expect("property_name must be a string")
            .clone()
            .expect("property_name cannot be null");
        let query_vector = args[1]
            .try_as_vector()
            .expect("query_vector must be a vector")
            .clone()
            .expect("query_vector cannot be null");
        let k = args[2]
            .try_as_uint32()
            .expect("k must be a uint32")
            .expect("k cannot be null");
        let l_value = args[3]
            .try_as_uint32()
            .expect("l_value must be a uint32")
            .expect("l_value cannot be null");
        let filter_condition = args[4].try_as_string().and_then(|s| s.clone());

        if k == 0 || l_value == 0 {
            return Err(anyhow::anyhow!("k and l_value must be positive").into());
        }
        if k > l_value {
            return Err(anyhow::anyhow!("l_value must be greater than or equal to k").into());
        }

        let current_graph = context
            .current_graph
            .as_ref()
            .expect("No current graph set");
        let graph_container = current_graph
            .as_any()
            .downcast_ref::<GraphContainer>()
            .expect("Failed to access GraphContainer");
        let GraphStorage::Memory(memory_graph) = graph_container.graph_storage();
        let graph_type = graph_container.graph_type();
        let property_id = resolve_property_name(&property_name, &*graph_type)
            .ok_or_else(|| anyhow::anyhow!("Property '{}' not found", property_name))?;

        let query_f32: Vec<f32> = query_vector.iter().map(|f| f.into_inner()).collect();
        if query_f32.is_empty() {
            return Err(anyhow::anyhow!("Query vector cannot be empty").into());
        }

        let filter_bitmap = if let Some(condition) = filter_condition {
            Some(generate_filter_bitmap(&context, memory_graph, &condition)?)
        } else {
            None
        };

        let search_results = memory_graph
            .vector_search(
                property_id,
                &query_f32,
                k as usize,
                l_value,
                filter_bitmap.as_ref(),
            )
            .map_err(|e| anyhow::anyhow!("Vector search failed: {}", e))?;

        let node_ids = Arc::new(UInt64Array::from(search_results));
        let chunk = data_chunk::DataChunk::new(vec![node_ids]);
        Ok(vec![chunk])
    })
}

/// Resolve property name to PropertyId by searching through vertex types
fn resolve_property_name(property_name: &str, graph_type: &dyn GraphTypeProvider) -> Option<u32> {
    for key in graph_type.vertex_type_keys() {
        if let Ok(Some(vertex_type)) = graph_type.get_vertex_type(&key) {
            if let Ok(Some((property_id, _))) = vertex_type.get_property(property_name) {
                return Some(property_id);
            }
        }
    }
    None
}

/// Generate filter bitmap from filter condition string
fn generate_filter_bitmap(
    _context: &SessionContext,
    _memory_graph: &Arc<MemoryGraph>,
    _filter_condition: &str,
) -> Result<BooleanArray, Box<dyn std::error::Error + Send + Sync>> {
    // TODO: Pass the pre-filter bitmap result to the storage layer
    todo!("Implement bitmap generation")
}
