use std::sync::Arc;

use minigu_catalog::provider::{GraphProvider, GraphTypeProvider, PropertiesProvider};
use minigu_common::data_chunk;
use minigu_common::data_type::{DataField, DataSchema, LogicalType};
use minigu_context::graph::{GraphContainer, GraphStorage};
use minigu_context::procedure::Procedure;

/// VectorSearch procedure for performing vector similarity search.
///
/// Function signature: CALL VectorSearch(property_name: String, query_vector: Vec<f32>, k: usize,
/// l_value: u32) Returns: A list of node IDs sorted by similarity (most similar first)
pub fn build_procedure() -> Procedure {
    let parameters = vec![
        LogicalType::String,    // property_name
        LogicalType::Vector(0), // query_vector (dimension validated at runtime)
        LogicalType::UInt32,    // k
        LogicalType::UInt32,    // l_value (optional, defaults to k*2)
    ];
    let schema = Arc::new(DataSchema::new(vec![DataField::new(
        "node_id".into(),
        LogicalType::UInt64,
        false,
    )]));

    Procedure::new(parameters, Some(schema), |context, args| {
        // Validate argument count and extract arguments
        assert_eq!(args.len(), 4);
        let property_name = args[0]
            .try_as_string()
            .expect("property_name must be a string")
            .as_ref()
            .expect("property_name cannot be null");
        let query_vector = args[1]
            .try_as_vector()
            .expect("query_vector must be a vector")
            .clone()
            .expect("query_vector cannot be null");
        let k = args[2]
            .try_as_uint32()
            .expect("k must be a uint32")
            .clone()
            .expect("k cannot be null");
        let l_value = args[3]
            .try_as_uint32()
            .expect("l_value must be a uint32")
            .clone()
            .expect("l_value cannot be null");

        // Validate parameters
        if k == 0 {
            return Err(anyhow::anyhow!("k must be positive, got {}", k).into());
        }
        if l_value == 0 {
            return Err(anyhow::anyhow!("l_value must be positive, got {}", l_value).into());
        }

        // Get current graph
        let current_graph = context
            .current_graph
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No current graph set"))?;
        let graph_container = current_graph
            .as_any()
            .downcast_ref::<GraphContainer>()
            .ok_or_else(|| anyhow::anyhow!("Failed to access GraphContainer"))?;
        let memory_graph = match graph_container.graph_storage() {
            GraphStorage::Memory(graph) => graph,
        };

        // Resolve property name to PropertyId
        let property_id = resolve_property_name(property_name, &*graph_container.graph_type())
            .ok_or_else(|| anyhow::anyhow!("Property '{}' not found", property_name))?;
        // Convert query vector from Vec<F32> to Vec<f32>
        let query_f32: Vec<f32> = query_vector.iter().map(|f| f.into_inner()).collect();
        // Perform vector search
        let search_results = memory_graph
            .vector_search(property_id, &query_f32, k as usize, l_value, None)
            .map_err(|e| anyhow::anyhow!("Vector search failed: {}", e))?;

        // Convert results to DataChunk
        use arrow::array::UInt64Array;
        let node_ids = Arc::new(UInt64Array::from(search_results));
        let chunk = data_chunk::DataChunk::new(vec![node_ids]);
        Ok(vec![chunk])
    })
}

/// Resolve property name to PropertyId by searching through vertex types
fn resolve_property_name(property_name: &str, graph_type: &dyn GraphTypeProvider) -> Option<u32> {
    // Search through all vertex types to find the property
    for key in graph_type.vertex_type_keys() {
        if let Ok(Some(vertex_type)) = graph_type.get_vertex_type(&key) {
            if let Ok(Some((property_id, _))) = vertex_type.get_property(property_name) {
                return Some(property_id);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use minigu_catalog::provider::ProcedureProvider;
    use minigu_common::value::F32;

    use super::*;

    #[test]
    fn test_vector_search_procedure_creation() {
        let procedure = build_procedure();

        // Check parameters
        let expected_params = vec![
            LogicalType::String,
            LogicalType::Vector(0),
            LogicalType::UInt32,
            LogicalType::UInt32,
        ];
        assert_eq!(procedure.parameters(), &expected_params);

        // Check schema
        assert!(procedure.schema().is_some());
        let schema = procedure.schema().unwrap();
        assert_eq!(schema.fields().len(), 1);
        assert_eq!(schema.fields()[0].name(), "node_id");
        assert_eq!(schema.fields()[0].ty(), &LogicalType::UInt64);
    }

    #[test]
    fn test_resolve_property_name_not_found() {
        use minigu_catalog::memory::graph_type::MemoryGraphTypeCatalog;

        let graph_type = MemoryGraphTypeCatalog::new();
        let result = resolve_property_name("nonexistent", &graph_type);

        assert!(result.is_none());
    }

    #[test]
    fn test_vector_search_parameter_types() {
        let procedure = build_procedure();

        // Test parameter types are correct
        let expected_params = vec![
            LogicalType::String,    // property_name
            LogicalType::Vector(0), // query_vector
            LogicalType::UInt32,    // k
            LogicalType::UInt32,    // l_value
        ];
        assert_eq!(procedure.parameters(), &expected_params);
    }

    #[test]
    fn test_vector_search_result_schema() {
        let procedure = build_procedure();

        // Test result schema is correct
        assert!(procedure.schema().is_some());
        let schema = procedure.schema().unwrap();
        assert_eq!(schema.fields().len(), 1);
        assert_eq!(schema.fields()[0].name(), "node_id");
        assert_eq!(schema.fields()[0].ty(), &LogicalType::UInt64);
        assert_eq!(schema.fields()[0].is_nullable(), false);
    }
}
