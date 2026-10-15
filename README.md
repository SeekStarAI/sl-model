# sl-model

`sl-model` 是 StarLight 的模型描述与格式适配层。它把外部模型容器转换为格式无关的 `ModelBundle`，并保留架构配置、张量形状、dtype、编码方式和权重字节视图。

本 crate 独立发布，不依赖 `sl-core`、`sl-graph`、`sl-device` 或 `sl-precision`。模型文件解析不直接生成执行命令；模型架构 lowering 与图编译属于后续独立阶段。

## 当前能力

- 定义带 schema 版本的 `ModelBundle`、模型 metadata、张量 dtype/encoding 和字节存储视图。
- Safetensors reader 以共享输入字节缓冲区创建张量视图，避免逐张量复制。
- 校验张量 shape、数据长度、分组量化载荷及 scale 长度。
- 保留 Safetensors 文件 metadata；架构和配置由调用方显式提供。

## 使用

```rust
use std::sync::Arc;

use serde_json::json;
use sl_model::{formats::safetensors, ModelMetadata};

let metadata = ModelMetadata::new("llama", json!({ "hidden_size": 4096 }))?;
let model_bytes: Arc<[u8]> = std::fs::read("model.safetensors")?.into();
let bundle = safetensors::load(model_bytes, metadata)?;
assert!(bundle.tensor("model.layers.0.self_attn.q_proj.weight").is_some());
```

## 格式路线

Safetensors 是首个 reader。GGUF 将作为下一种格式接入，并保留其 block quantization 编码；PTH/PyTorch pickle 不会在推理进程内直接反序列化，优先通过隔离的导出工具转为安全格式。

模型格式契约、依赖边界和测试要求见 [DEVELOPMENT.md](DEVELOPMENT.md)。
