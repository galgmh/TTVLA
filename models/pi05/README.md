# models/pi05 — π0.5 模型层（项目自有子包，纯 PyTorch）

- 来源：OpenPI 官方提交 `215abfb217dbac7d5f1273282331b9b1866c0479`
  （https://github.com/Physical-Intelligence/openpi）
- 改写：import 路径 `openpi.*` → `models.pi05.*`；官方 JAX/Flax 实现
  （`pi0.py`、`model.py`、`gemma.py`、`siglip.py`、`lora.py`、`sharding.py`、
  `array_typing.py`、`nnx_utils.py`）已全部删除，`gemma.py` 的 Gemma 结构配置
  并入 `pi0_config.py`；训练/微调/评测只走 PyTorch。
- 这是项目中 π0.5 的**唯一实现**（不再有 third_party 副本）；训练/微调代码
  在本项目自己的 `training/pi05/` 里。
- `thermal/`：Thermal-π0.5 热觉分支（本项目的增量实现），说明见
  `models/pi05/THERMAL_PI05.md`，规格见
  `models/pi05/INTEGRATED_PI05_ARCHITECTURE.md`。

## 文件对应关系

| 本目录 | openpi 原路径 | 本项目改动 |
|---|---|---|
| `pi0_pytorch.py` | `src/openpi/models_pytorch/pi0_pytorch.py` | import 改写 + 热觉编码器接入 |
| `gemma_pytorch.py` | `src/openpi/models_pytorch/gemma_pytorch.py` | `projection_dim` 改为 `width`（兼容 dummy），dummy 视觉塔缩小 |
| `preprocessing_pytorch.py` | `src/openpi/models_pytorch/preprocessing_pytorch.py` | 预处理结果保留热觉字段（None 时行为不变） |
| `observation.py` | `src/openpi/models/model.py`（仅 `Observation`） | Flax struct/jaxtyping → 纯 dataclass，增加 13 个热觉字段 |
| `pi0_config.py` | `src/openpi/models/pi0_config.py` + `gemma.py` 的配置部分 | 删除 JAX 专用接口与 Flax 模块 |
| `image_tools.py` | `src/openpi/shared/image_tools.py`（仅 PyTorch 部分） | 删除 JAX 版 `resize_with_pad` |
| `thermal/` | **无（本项目新增）** | 三个核心模块（接触编码器 / 空间记忆 / 主动策略）+ 输出结构、损失与 π0.5 接入；训练/数据侧工具在 `training/pi05/thermal/` |
| `transformers_replace/` | `src/openpi/models_pytorch/transformers_replace/` | 原样保留的 HuggingFace 补丁 |

## 运行环境

用项目根目录的 `.venv`（torch 2.7.1+cu126、transformers 4.53.2、
numpy 1.26.x），`transformers_replace` 已安装到 site-packages；**不再需要
JAX/Flax/Orbax**。若重建环境：

```bash
SP=$(python -c "import transformers, os; print(os.path.dirname(transformers.__file__))")
cp -r models/pi05/transformers_replace/. "$SP"
```

注意：cu126 轮子不支持 RTX 5060（sm_120），本机 GPU 训练需换 cu128 构建；
训练/闭环评估按项目约定在云端 GPU 机器上跑。

## 使用

```python
from models.pi05 import PI0Pytorch, Pi0Config, ThermalPi0Config

config = Pi0Config(pi05=True, thermal=ThermalPi0Config(enabled=True))
model = PI0Pytorch(config)
```

`thermal=None` 或 `enabled=False` 时与原始 π0.5 数值完全一致（Stage-0 回归
`STAGE0_IDENTICAL`）。训练阶段、数据格式与训练入口见
`models/pi05/THERMAL_PI05.md`。

## 测试

```bash
cd ~/ThermTacVLA
PYTHONPATH=. .venv/bin/python -m pytest models/pi05/thermal -q
```
