// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! Format-independent model bundles and model format readers for StarLight.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

pub mod formats;

/// Version of the in-memory model bundle contract.
pub const MODEL_BUNDLE_SCHEMA_VERSION: u32 = 1;

/// Errors raised while validating model metadata, tensor storage, or formats.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelError {
    InvalidArchitecture,
    InvalidTensorName,
    InvalidDataRange,
    ShapeOverflow,
    InvalidGroupSize,
    InvalidQuantizationScheme,
    PayloadLengthMismatch { expected: usize, actual: usize },
    ScaleLengthMismatch { expected: usize, actual: usize },
    DuplicateTensor(String),
    UnsupportedDType(String),
    Format(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArchitecture => formatter.write_str("architecture name must not be empty"),
            Self::InvalidTensorName => formatter.write_str("tensor name must not be empty"),
            Self::InvalidDataRange => {
                formatter.write_str("tensor data range is outside its backing buffer")
            }
            Self::ShapeOverflow => {
                formatter.write_str("tensor element count or byte size overflows usize")
            }
            Self::InvalidGroupSize => {
                formatter.write_str("quantization group size must be greater than zero")
            }
            Self::InvalidQuantizationScheme => {
                formatter.write_str("quantization scheme must not be empty")
            }
            Self::PayloadLengthMismatch { expected, actual } => {
                write!(
                    formatter,
                    "tensor payload has {actual} bytes; expected {expected}"
                )
            }
            Self::ScaleLengthMismatch { expected, actual } => {
                write!(
                    formatter,
                    "scale payload has {actual} bytes; expected {expected}"
                )
            }
            Self::DuplicateTensor(name) => write!(formatter, "tensor `{name}` already exists"),
            Self::UnsupportedDType(dtype) => write!(formatter, "unsupported tensor dtype: {dtype}"),
            Self::Format(message) => formatter.write_str(message),
        }
    }
}

impl Error for ModelError {}

/// Common tensor element types accepted by model format adapters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorDType {
    Bool,
    F4,
    F6E2M3,
    F6E3M2,
    U8,
    I8,
    F8E5M2,
    F8E4M3,
    F8E8M0,
    F8E4M3Fnuz,
    F8E5M2Fnuz,
    I16,
    U16,
    F16,
    BF16,
    I32,
    U32,
    F32,
    F64,
    I64,
    U64,
}

impl TensorDType {
    /// Returns the dtype's storage width in bits.
    pub fn bit_width(self) -> usize {
        match self {
            Self::F4 => 4,
            Self::F6E2M3 | Self::F6E3M2 => 6,
            Self::Bool
            | Self::U8
            | Self::I8
            | Self::F8E5M2
            | Self::F8E4M3
            | Self::F8E8M0
            | Self::F8E4M3Fnuz
            | Self::F8E5M2Fnuz => 8,
            Self::I16 | Self::U16 | Self::F16 | Self::BF16 => 16,
            Self::I32 | Self::U32 | Self::F32 => 32,
            Self::F64 | Self::I64 | Self::U64 => 64,
        }
    }
}

/// Quantized integer formats understood by generic tensor storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuantizationFormat {
    Int4,
    Int8,
}

impl QuantizationFormat {
    fn bit_width(self) -> usize {
        match self {
            Self::Int4 => 4,
            Self::Int8 => 8,
        }
    }
}

/// A byte range backed by shared, owned storage.
#[derive(Clone, Eq, PartialEq)]
pub struct TensorData {
    backing: Arc<[u8]>,
    range: Range<usize>,
}

impl TensorData {
    /// Creates storage by taking ownership of a byte vector.
    pub fn from_vec(bytes: Vec<u8>) -> Self {
        let backing: Arc<[u8]> = bytes.into();
        let range = 0..backing.len();
        Self { backing, range }
    }

    /// Creates a checked view into shared storage without copying its bytes.
    pub fn from_shared(backing: Arc<[u8]>, range: Range<usize>) -> Result<Self, ModelError> {
        if range.start > range.end || range.end > backing.len() {
            return Err(ModelError::InvalidDataRange);
        }
        Ok(Self { backing, range })
    }

    /// Returns the bytes represented by this view.
    pub fn as_bytes(&self) -> &[u8] {
        &self.backing[self.range.clone()]
    }

    /// Returns the length of this view in bytes.
    pub fn len(&self) -> usize {
        self.range.len()
    }

    /// Returns whether this view contains no bytes.
    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }

    /// Returns the byte range within the shared backing allocation.
    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }

    /// Returns the shared backing allocation.
    pub fn backing(&self) -> &Arc<[u8]> {
        &self.backing
    }
}

impl fmt::Debug for TensorData {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TensorData")
            .field("range", &self.range)
            .field("backing_len", &self.backing.len())
            .finish()
    }
}

/// Describes how a tensor's payload bytes should be interpreted.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TensorEncoding {
    Dense,
    GroupwiseQuantized {
        format: QuantizationFormat,
        group_size: usize,
        scales: TensorData,
    },
    OpaqueQuantized {
        scheme: String,
    },
}

/// A tensor's logical shape, dtype, encoding, and payload view.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelTensor {
    shape: Vec<usize>,
    dtype: TensorDType,
    encoding: TensorEncoding,
    payload: TensorData,
}

impl ModelTensor {
    /// Creates a tensor and validates its storage against the logical shape.
    pub fn new(
        shape: Vec<usize>,
        dtype: TensorDType,
        encoding: TensorEncoding,
        payload: TensorData,
    ) -> Result<Self, ModelError> {
        let element_count = element_count(&shape)?;
        match &encoding {
            TensorEncoding::Dense => {
                let bit_count = element_count
                    .checked_mul(dtype.bit_width())
                    .ok_or(ModelError::ShapeOverflow)?;
                let expected = bit_count.checked_add(7).ok_or(ModelError::ShapeOverflow)? / 8;
                if payload.len() != expected {
                    return Err(ModelError::PayloadLengthMismatch {
                        expected,
                        actual: payload.len(),
                    });
                }
            }
            TensorEncoding::GroupwiseQuantized {
                format,
                group_size,
                scales,
            } => {
                if *group_size == 0 {
                    return Err(ModelError::InvalidGroupSize);
                }
                let bit_count = element_count
                    .checked_mul(format.bit_width())
                    .ok_or(ModelError::ShapeOverflow)?;
                let expected_payload =
                    bit_count.checked_add(7).ok_or(ModelError::ShapeOverflow)? / 8;
                if payload.len() != expected_payload {
                    return Err(ModelError::PayloadLengthMismatch {
                        expected: expected_payload,
                        actual: payload.len(),
                    });
                }
                let group_count =
                    element_count / group_size + usize::from(element_count % group_size != 0);
                let expected_scales = group_count
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or(ModelError::ShapeOverflow)?;
                if scales.len() != expected_scales {
                    return Err(ModelError::ScaleLengthMismatch {
                        expected: expected_scales,
                        actual: scales.len(),
                    });
                }
            }
            TensorEncoding::OpaqueQuantized { scheme } => {
                if scheme.trim().is_empty() {
                    return Err(ModelError::InvalidQuantizationScheme);
                }
            }
        }

        Ok(Self {
            shape,
            dtype,
            encoding,
            payload,
        })
    }

    /// Returns the logical shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the logical dtype.
    pub fn dtype(&self) -> TensorDType {
        self.dtype
    }

    /// Returns the storage encoding.
    pub fn encoding(&self) -> &TensorEncoding {
        &self.encoding
    }

    /// Returns the packed or dense payload view.
    pub fn payload(&self) -> &TensorData {
        &self.payload
    }

    /// Returns the logical element count.
    pub fn element_count(&self) -> Result<usize, ModelError> {
        element_count(&self.shape)
    }
}

/// Model-level metadata shared by format adapters and graph lowering.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelMetadata {
    architecture: String,
    config: serde_json::Value,
    source_format: Option<String>,
    properties: BTreeMap<String, String>,
}

impl ModelMetadata {
    /// Creates model metadata with an explicit architecture and configuration.
    pub fn new(
        architecture: impl Into<String>,
        config: serde_json::Value,
    ) -> Result<Self, ModelError> {
        let architecture = architecture.into();
        if architecture.trim().is_empty() {
            return Err(ModelError::InvalidArchitecture);
        }
        Ok(Self {
            architecture,
            config,
            source_format: None,
            properties: BTreeMap::new(),
        })
    }

    /// Returns the model architecture identifier.
    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    /// Returns the architecture-specific configuration document.
    pub fn config(&self) -> &serde_json::Value {
        &self.config
    }

    /// Returns the source format identifier, if known.
    pub fn source_format(&self) -> Option<&str> {
        self.source_format.as_deref()
    }

    /// Returns preserved source-format properties.
    pub fn properties(&self) -> &BTreeMap<String, String> {
        &self.properties
    }

    /// Adds a source-format property without discarding existing metadata.
    pub fn insert_property(&mut self, key: String, value: String) {
        self.properties.insert(key, value);
    }

    pub(crate) fn set_source_format(&mut self, source_format: impl Into<String>) {
        self.source_format = Some(source_format.into());
    }
}

/// A validated, format-independent model description and tensor catalog.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelBundle {
    schema_version: u32,
    metadata: ModelMetadata,
    tensors: BTreeMap<String, ModelTensor>,
}

impl ModelBundle {
    /// Creates an empty bundle using the current schema version.
    pub fn new(metadata: ModelMetadata) -> Self {
        Self {
            schema_version: MODEL_BUNDLE_SCHEMA_VERSION,
            metadata,
            tensors: BTreeMap::new(),
        }
    }

    /// Returns the schema version.
    pub fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Returns model-level metadata.
    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    /// Returns the tensor catalog.
    pub fn tensors(&self) -> &BTreeMap<String, ModelTensor> {
        &self.tensors
    }

    /// Looks up a tensor by its model-level name.
    pub fn tensor(&self, name: &str) -> Option<&ModelTensor> {
        self.tensors.get(name)
    }

    /// Adds a uniquely named tensor to the bundle.
    pub fn insert_tensor(
        &mut self,
        name: impl Into<String>,
        tensor: ModelTensor,
    ) -> Result<(), ModelError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(ModelError::InvalidTensorName);
        }
        if self.tensors.contains_key(&name) {
            return Err(ModelError::DuplicateTensor(name));
        }
        self.tensors.insert(name, tensor);
        Ok(())
    }
}

fn element_count(shape: &[usize]) -> Result<usize, ModelError> {
    shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
        .ok_or(ModelError::ShapeOverflow)
}

#[cfg(test)]
mod tests {
    use super::{
        ModelBundle, ModelError, ModelMetadata, ModelTensor, QuantizationFormat, TensorDType,
        TensorData, TensorEncoding,
    };
    use serde_json::json;
    use std::sync::Arc;

    #[test]
    fn validates_dense_tensor_shape_and_payload() {
        let tensor = ModelTensor::new(
            vec![2, 3],
            TensorDType::F32,
            TensorEncoding::Dense,
            TensorData::from_vec(vec![0; 24]),
        )
        .unwrap();

        assert_eq!(tensor.element_count().unwrap(), 6);
        assert_eq!(
            ModelTensor::new(
                vec![2, 3],
                TensorDType::F32,
                TensorEncoding::Dense,
                TensorData::from_vec(vec![0; 20]),
            )
            .err(),
            Some(ModelError::PayloadLengthMismatch {
                expected: 24,
                actual: 20,
            })
        );
    }

    #[test]
    fn validates_quantized_payload_and_scale_lengths() {
        let tensor = ModelTensor::new(
            vec![5],
            TensorDType::F32,
            TensorEncoding::GroupwiseQuantized {
                format: QuantizationFormat::Int4,
                group_size: 4,
                scales: TensorData::from_vec(vec![0; 8]),
            },
            TensorData::from_vec(vec![0; 3]),
        )
        .unwrap();

        assert_eq!(tensor.element_count().unwrap(), 5);
        assert_eq!(tensor.payload().len(), 3);
    }

    #[test]
    fn bundle_keeps_shared_tensor_storage() {
        let backing: Arc<[u8]> = vec![0, 1, 2, 3, 4, 5].into();
        let tensor_data = TensorData::from_shared(backing.clone(), 2..6).unwrap();
        let tensor = ModelTensor::new(
            vec![1],
            TensorDType::U32,
            TensorEncoding::Dense,
            tensor_data,
        )
        .unwrap();
        let metadata = ModelMetadata::new("test", json!({})).unwrap();
        let mut bundle = ModelBundle::new(metadata);
        bundle.insert_tensor("weight", tensor).unwrap();

        let stored = bundle.tensor("weight").unwrap().payload();
        assert!(Arc::ptr_eq(stored.backing(), &backing));
        assert_eq!(stored.as_bytes(), &[2, 3, 4, 5]);
    }

    #[test]
    fn rejects_duplicate_tensor_names_and_empty_architecture() {
        assert_eq!(
            ModelMetadata::new("  ", json!({})).err(),
            Some(ModelError::InvalidArchitecture)
        );
        let metadata = ModelMetadata::new("test", json!({})).unwrap();
        let tensor = ModelTensor::new(
            vec![1],
            TensorDType::U8,
            TensorEncoding::Dense,
            TensorData::from_vec(vec![1]),
        )
        .unwrap();
        let mut bundle = ModelBundle::new(metadata);
        bundle.insert_tensor("weight", tensor.clone()).unwrap();

        assert_eq!(
            bundle.insert_tensor("weight", tensor),
            Err(ModelError::DuplicateTensor("weight".to_string()))
        );
    }
}
