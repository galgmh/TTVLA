"""接触事件状态机与固定容量热空间记忆库。

MemoryBank是推理会话状态，不是传感器Observation，也不是可训练参数。一次持续
接触只在结束时写入一次；当前进行中的接触以临时token参与本次prefix构造。
"""

from __future__ import annotations

import torch


class ThermalMemoryBank:
    """按batch独立维护最多K个完成接触token，满时FIFO淘汰。"""

    def __init__(
        self,
        capacity: int,
        token_dim: int,
        *,
        on_threshold: float,
        off_threshold: float,
        on_steps: int,
        off_steps: int,
        momentum: float,
    ):
        self.capacity = capacity
        self.token_dim = token_dim
        self.on_threshold = on_threshold
        self.off_threshold = off_threshold
        self.on_steps = on_steps
        self.off_steps = off_steps
        self.momentum = momentum
        self.clear()

    def clear(self) -> None:
        self.tokens = None
        self.mask = None
        self.active = None
        self.active_token = None
        self.on_count = None
        self.off_count = None

    def reset(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> None:
        self.tokens = torch.zeros(batch_size, self.capacity, self.token_dim, device=device, dtype=dtype)
        self.mask = torch.zeros(batch_size, self.capacity, device=device, dtype=torch.bool)
        self.active = torch.zeros(batch_size, device=device, dtype=torch.bool)
        self.active_token = torch.zeros(batch_size, self.token_dim, device=device, dtype=dtype)
        self.on_count = torch.zeros(batch_size, device=device, dtype=torch.long)
        self.off_count = torch.zeros(batch_size, device=device, dtype=torch.long)

    def _ensure(self, token: torch.Tensor) -> None:
        if (
            self.tokens is None
            or self.tokens.shape[0] != token.shape[0]
            or self.tokens.device != token.device
            or self.tokens.dtype != token.dtype
        ):
            self.reset(token.shape[0], device=token.device, dtype=token.dtype)

    def reset_rows(self, reset_mask: torch.Tensor) -> None:
        """仅清空新episode对应的batch行。"""
        if self.tokens is None:
            return
        rows = reset_mask.to(device=self.tokens.device, dtype=torch.bool)
        self.tokens[rows] = 0
        self.mask[rows] = False
        self.active[rows] = False
        self.active_token[rows] = 0
        self.on_count[rows] = 0
        self.off_count[rows] = 0

    def _append(self, row: int, token: torch.Tensor) -> None:
        valid = int(self.mask[row].sum().item())
        if valid < self.capacity:
            slot = valid
        else:
            self.tokens[row, :-1] = self.tokens[row, 1:].clone()
            self.mask[row, :-1] = self.mask[row, 1:].clone()
            slot = self.capacity - 1
        self.tokens[row, slot] = token
        self.mask[row, slot] = True

    @torch.no_grad()
    def update(self, current_token: torch.Tensor, valid_probability: torch.Tensor) -> None:
        """更新事件状态；token被detach，避免跨控制周期保留计算图。"""
        token = current_token.detach()
        probability = valid_probability.detach().reshape(-1)
        self._ensure(token)
        for row in range(token.shape[0]):
            if not bool(self.active[row]):
                self.on_count[row] = self.on_count[row] + 1 if probability[row] >= self.on_threshold else 0
                if self.on_count[row] >= self.on_steps:
                    self.active[row] = True
                    self.active_token[row] = token[row]
                    self.on_count[row] = 0
                    self.off_count[row] = 0
                continue

            if probability[row] >= self.off_threshold:
                self.active_token[row].mul_(self.momentum).add_(token[row], alpha=1.0 - self.momentum)
                self.off_count[row] = 0
            else:
                self.off_count[row] += 1
                if self.off_count[row] >= self.off_steps:
                    self._append(row, self.active_token[row])
                    self.active[row] = False
                    self.active_token[row].zero_()
                    self.off_count[row] = 0

    def read_with_current(
        self,
        current_token: torch.Tensor,
        valid_probability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回长期记忆和临时当前接触；预激活阶段也允许当前token进入prefix。"""
        self._ensure(current_token)
        present = self.active | (valid_probability.detach().reshape(-1) >= self.off_threshold)
        current = torch.where(present[:, None], current_token, torch.zeros_like(current_token))
        return (
            torch.cat([self.tokens, current[:, None]], dim=1),
            torch.cat([self.mask, present[:, None]], dim=1),
        )


__all__ = ["ThermalMemoryBank"]
