# sl-model Development Guide

## Responsibility

`sl-model` owns model-level metadata, tensor catalogs, storage encodings, and external model format adapters. It does not own inference scheduling, graph optimization, device execution, or numeric quantization algorithms.

The dependency direction is intentionally one-way:

```text
external format -> sl-model ModelBundle -> model-family lowering -> sl-graph
                                               |                       |
                                      sl-precision transforms      MemoryPlan
                                                                       |
                                                                  sl-core build
```

Format readers must not depend on `sl-core` or emit `CommandBuffer`/`MemoryPlan`. `ModelBundle` is an interchange boundary; graph lowering can evolve without rewriting file readers. `sl-model` must remain independently usable by conversion and inspection tools.

## ModelBundle Contract

- `MODEL_BUNDLE_SCHEMA_VERSION` versions the in-memory contract. Serialized model packages must carry their own explicit version before a writer is added.
- Architecture name and model configuration are explicit inputs; readers must not infer a model family from tensor names alone.
- Tensor shapes describe logical elements. A scalar has an empty shape and one element; any zero dimension yields an empty tensor.
- Dense payload sizes are checked against shape and dtype bit width.
- Group-wise quantized tensors keep packed payload and F32 scale bytes as separate views, with format and group size in the encoding descriptor.
- A `TensorData` owns a shared `Arc<[u8]>` plus a checked byte range. Readers should retain shared backing storage instead of copying individual tensors where possible.
- Source-format metadata is preserved as opaque string properties. Format-specific details must not leak into generic graph or device types.

## Format Adapter Rules

- Parse and validate the container before exposing any tensor views.
- Preserve source dtype, shape, tensor names, and metadata exactly unless a documented conversion is explicitly requested.
- Unknown format-specific quantization encodings must be preserved as opaque payloads or rejected with a specific error; never silently reinterpret them as dense floats.
- Safetensors contains tensor data, not a complete model definition. Its reader therefore requires caller-provided architecture/configuration metadata.
- GGUF support must preserve block quantization parameters and tokenizer/model metadata.
- PTH files may contain executable pickle payloads. Do not load untrusted PTH data in the inference process; use an isolated conversion/export path.

## Validation

Run from the workspace root:

```sh
cargo test -p sl-model
cargo clippy -p sl-model --all-targets -- -D warnings
```

Reader tests should cover malformed containers, dtype mapping, metadata retention, zero-copy backing ranges, and shape/payload mismatch rejection. Every new format needs fixture-based tests and a documented support matrix.

## Planned Work

1. Add GGUF parsing with round-trip preservation of known quantized tensor encodings.
2. Define a versioned on-disk StarLight model package only after the in-memory contract stabilizes.
3. Add model-family lowering adapters that map `ModelBundle` into generic `sl-graph` input.
4. Move model-file quantization orchestration and the Safetensors CLI out of `sl-precision`; keep numeric packing APIs there.
5. Add sharded Safetensors index/config loading and streaming for checkpoints larger than memory.