//! Zero-copy reader for Safetensors files.

use crate::{
    ModelBundle, ModelError, ModelMetadata, ModelTensor, TensorDType, TensorData, TensorEncoding,
};
use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;
use std::ops::Range;
use std::sync::Arc;

/// Loads a Safetensors file from shared bytes into a model bundle.
///
/// Safetensors does not define the model architecture or its configuration;
/// callers must provide those through `metadata`.
pub fn load(bytes: Arc<[u8]>, mut metadata: ModelMetadata) -> Result<ModelBundle, ModelError> {
    let (_, source_metadata) = SafeTensors::read_metadata(&bytes)
        .map_err(|error| ModelError::Format(error.to_string()))?;
    let file =
        SafeTensors::deserialize(&bytes).map_err(|error| ModelError::Format(error.to_string()))?;

    metadata.set_source_format("safetensors");
    if let Some(properties) = source_metadata.metadata() {
        for (key, value) in properties {
            if !metadata.properties().contains_key(key) {
                metadata.insert_property(key.clone(), value.clone());
            }
        }
    }

    let mut bundle = ModelBundle::new(metadata);
    for (name, tensor) in file.tensors() {
        let dtype = map_dtype(tensor.dtype())?;
        let range = tensor_range(&bytes, &tensor)?;
        let payload = TensorData::from_shared(bytes.clone(), range)?;
        let model_tensor = ModelTensor::new(
            tensor.shape().to_vec(),
            dtype,
            TensorEncoding::Dense,
            payload,
        )?;
        bundle.insert_tensor(name, model_tensor)?;
    }

    Ok(bundle)
}

fn map_dtype(dtype: Dtype) -> Result<TensorDType, ModelError> {
    let mapped = match dtype {
        Dtype::BOOL => TensorDType::Bool,
        Dtype::F4 => TensorDType::F4,
        Dtype::F6_E2M3 => TensorDType::F6E2M3,
        Dtype::F6_E3M2 => TensorDType::F6E3M2,
        Dtype::U8 => TensorDType::U8,
        Dtype::I8 => TensorDType::I8,
        Dtype::F8_E5M2 => TensorDType::F8E5M2,
        Dtype::F8_E4M3 => TensorDType::F8E4M3,
        Dtype::F8_E8M0 => TensorDType::F8E8M0,
        Dtype::I16 => TensorDType::I16,
        Dtype::U16 => TensorDType::U16,
        Dtype::F16 => TensorDType::F16,
        Dtype::BF16 => TensorDType::BF16,
        Dtype::I32 => TensorDType::I32,
        Dtype::U32 => TensorDType::U32,
        Dtype::F32 => TensorDType::F32,
        Dtype::F64 => TensorDType::F64,
        Dtype::I64 => TensorDType::I64,
        Dtype::U64 => TensorDType::U64,
        _ => return Err(ModelError::UnsupportedDType(dtype.to_string())),
    };
    Ok(mapped)
}

fn tensor_range(bytes: &[u8], tensor: &TensorView<'_>) -> Result<Range<usize>, ModelError> {
    let base_address = bytes.as_ptr() as usize;
    let tensor_address = tensor.data().as_ptr() as usize;
    let start = tensor_address
        .checked_sub(base_address)
        .ok_or(ModelError::InvalidDataRange)?;
    let end = start
        .checked_add(tensor.data().len())
        .ok_or(ModelError::InvalidDataRange)?;
    if end > bytes.len() {
        return Err(ModelError::InvalidDataRange);
    }
    Ok(start..end)
}

#[cfg(test)]
mod tests {
    use super::load;
    use crate::{ModelError, ModelMetadata, TensorDType};
    use safetensors::tensor::{serialize, Dtype, TensorView};
    use serde_json::json;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn reads_safetensors_metadata_and_shares_tensor_bytes() {
        let weight_bytes: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let token_bytes = [4_i32, 9_i32]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let tensors = HashMap::from([
            (
                "weight".to_string(),
                TensorView::new(Dtype::F32, vec![2], &weight_bytes).unwrap(),
            ),
            (
                "token_ids".to_string(),
                TensorView::new(Dtype::I32, vec![2], &token_bytes).unwrap(),
            ),
        ]);
        let properties = HashMap::from([("format_tag".to_string(), "unit-test".to_string())]);
        let serialized = serialize(tensors, Some(properties)).unwrap();
        let bytes: Arc<[u8]> = serialized.into();
        let metadata = ModelMetadata::new("test-architecture", json!({ "width": 2 })).unwrap();
        let bundle = load(bytes.clone(), metadata).unwrap();

        let weight = bundle.tensor("weight").unwrap();
        assert_eq!(weight.dtype(), TensorDType::F32);
        assert_eq!(weight.shape(), &[2]);
        assert_eq!(weight.payload().as_bytes(), weight_bytes);
        assert!(Arc::ptr_eq(weight.payload().backing(), &bytes));
        assert_eq!(bundle.metadata().source_format(), Some("safetensors"));
        assert_eq!(
            bundle.metadata().properties().get("format_tag").unwrap(),
            "unit-test"
        );
        assert_eq!(
            bundle.tensor("token_ids").unwrap().dtype(),
            TensorDType::I32
        );
    }

    #[test]
    fn rejects_invalid_safetensors_input() {
        let metadata = ModelMetadata::new("test-architecture", json!({})).unwrap();
        assert!(matches!(
            load(Arc::<[u8]>::from(&b"bad"[..]), metadata),
            Err(ModelError::Format(_))
        ));
    }
}
