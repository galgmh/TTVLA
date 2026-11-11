"""Thermal-π0.5 分阶段冻结规则（规格第 9 节）。

阶段划分：

1. 冻结 π0.5 全部参数，只训练热接触编码器、接触 token 构造、候选点编码器、
   空间记忆（含温度/安全头）；
2. 冻结 π0.5 与阶段 1 模块，只训练 ``ActiveStrategyModule``（技能与目标点）；
3. 训练全部热觉模块、thermal→prefix 投影与 Action Expert（含动作投影），
   冻结 SigLIP 与 PaliGemma VLM 主干；
4. 联合微调：在阶段 3 基础上解冻 PaliGemma 最后 ``paligemma_last_layers`` 层。

说明：PyTorch 版 openpi 目前没有 LoRA 实现（``openpi/models/lora.py`` 只服务
JAX 路径），因此阶段 4 用「最后 N 层全量微调」代替规格中的 PaliGemma LoRA；
学习率仍按规格分三组：新模块 1e-4、Action Expert 5e-5、PaliGemma 1e-5。

冻结只改 ``requires_grad``，不动 ``state_dict``：任何阶段保存的 checkpoint 都能
被未冻结/已冻结的模型按 ``load_state_dict_compat`` 原样加载。
"""

from __future__ import annotations

from torch import nn

#: 规格允许的训练阶段。
STAGES = (1, 2, 3, 4)

_THERMAL_PREFIX = "thermal_encoder."
#: 阶段 1 可训练的集成编码器子模块（温度/安全头位于 spatial_memory 内）。
_STAGE1_MODULES = ("contact_encoder", "contact_builder", "point_encoder", "spatial_memory")
#: 阶段 3 起随 Action Expert 一起训练的动作投影层。
_ACTION_MODULES = (
    "action_in_proj",
    "action_out_proj",
    "time_mlp_in",
    "time_mlp_out",
    "state_proj",
    "action_time_mlp_in",
    "action_time_mlp_out",
)

#: 每个阶段的损失项（``compute_loss_dict`` 的键）；用于分阶段只回传相关项。
_STAGE_LOSS_KEYS = {
    1: ("contact_temp", "contact_valid", "field_temp", "field_safe"),
    2: ("skill", "target"),
    3: ("flow", "contact_temp", "contact_valid", "field_temp", "field_safe", "skill", "target"),
    4: ("flow", "contact_temp", "contact_valid", "field_temp", "field_safe", "skill", "target"),
}


def _validate_stage(stage: int) -> None:
    if stage not in STAGES:
        raise ValueError(f"stage 必须是 {STAGES} 之一，得到 {stage!r}")


def _set_requires_grad(module: nn.Module, *, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(value)


def _thermal_encoder(model: nn.Module) -> nn.Module:
    thermal = getattr(model, "thermal_encoder", None)
    if thermal is None:
        raise ValueError("模型没有 thermal_encoder：请先用 ThermalPi0Config(enabled=True) 构建模型")
    return thermal


def _paligemma_language_layers(model: nn.Module) -> nn.ModuleList:
    """返回 PaliGemma 语言塔的逐层列表（不同 transformers 版本路径一致）。"""
    paligemma = model.paligemma_with_expert.paligemma
    return paligemma.model.language_model.layers


def configure_stage(model: nn.Module, stage: int, *, paligemma_last_layers: int = 4) -> dict[str, int]:
    """按阶段设置 ``requires_grad``，返回各角色可训练参数个数（便于日志）。

    角色：``thermal`` / ``action_expert`` / ``paligemma``。未列出的参数全部冻结。
    """
    _validate_stage(stage)
    if stage == 4 and paligemma_last_layers < 1:
        raise ValueError("stage=4 时 paligemma_last_layers 必须 >= 1")
    thermal = _thermal_encoder(model)

    # 先全部冻结，再按阶段解冻，避免调用方残留的 requires_grad 影响结果。
    _set_requires_grad(model, value=False)

    if stage == 1:
        for name in _STAGE1_MODULES:
            _set_requires_grad(getattr(thermal, name), value=True)
    elif stage == 2:
        _set_requires_grad(thermal.strategy, value=True)
    else:
        _set_requires_grad(thermal, value=True)
        _set_requires_grad(model.paligemma_with_expert.gemma_expert, value=True)
        for name in _ACTION_MODULES:
            module = getattr(model, name, None)
            if module is not None:
                _set_requires_grad(module, value=True)
        if stage == 4:
            layers = _paligemma_language_layers(model)
            count = min(paligemma_last_layers, len(layers))
            for layer in layers[-count:]:
                _set_requires_grad(layer, value=True)

    counts = {"thermal": 0, "action_expert": 0, "paligemma": 0}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            counts[_parameter_role(name)] += 1
    return counts


def _parameter_role(name: str) -> str:
    """把参数名归入三组学习率角色（与规格第 9 节一致）。"""
    if name.startswith(_THERMAL_PREFIX):
        return "thermal"
    if name.startswith("paligemma_with_expert.gemma_expert."):
        return "action_expert"
    if name.split(".", 1)[0] in set(_ACTION_MODULES):
        # 顶层动作投影：action_in_proj.weight 等。
        return "action_expert"
    return "paligemma"


def parameter_groups(
    model: nn.Module,
    stage: int | None = None,
    *,
    thermal_lr: float = 1e-4,
    action_expert_lr: float = 5e-5,
    paligemma_lr: float = 1e-5,
) -> list[dict]:
    """按规格学习率把当前可训练参数分成优化器参数组。

    需先调用 ``configure_stage`` 设置 ``requires_grad``；``stage`` 仅用于校验，
    不会改变冻结状态。返回空组会被省略。
    """
    if stage is not None:
        _validate_stage(stage)
    learning_rates = {"thermal": thermal_lr, "action_expert": action_expert_lr, "paligemma": paligemma_lr}
    buckets: dict[str, list[nn.Parameter]] = {role: [] for role in learning_rates}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            buckets[_parameter_role(name)].append(parameter)
    groups = [
        {"name": role, "params": params, "lr": learning_rates[role]}
        for role, params in buckets.items()
        if params
    ]
    if not groups:
        raise ValueError("没有可训练参数：请先调用 configure_stage(model, stage)")
    return groups


def stage_loss_keys(stage: int) -> tuple[str, ...]:
    """返回该阶段应参与反向传播的损失项（其余项只记日志）。"""
    _validate_stage(stage)
    return _STAGE_LOSS_KEYS[stage]


def trainable_parameter_names(model: nn.Module) -> list[str]:
    """便于测试/日志：当前 requires_grad=True 的参数名。"""
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


__all__ = [
    "STAGES",
    "configure_stage",
    "parameter_groups",
    "stage_loss_keys",
    "trainable_parameter_names",
]
