"""PaliGemma SentencePiece 分词器（pi05 离散状态输入的文本格式）。

参考 openpi `src/openpi/models/tokenizer.py` 的 `PaligemmaTokenizer` 重写：
不再通过 `openpi.shared.download` 访问 GCS，而是要求本地已有
`paligemma_tokenizer.model`（默认用 openpi 缓存的下载结果）。
"""

from __future__ import annotations

import logging
import os
import pathlib

import numpy as np
import sentencepiece

logger = logging.getLogger(__name__)

#: 默认 tokenizer 文件位置：优先环境变量，其次 openpi 的下载缓存。
DEFAULT_TOKENIZER_PATH = pathlib.Path(
    os.environ.get(
        "PALIGEMMA_TOKENIZER_PATH",
        "~/.cache/openpi/big_vision/paligemma_tokenizer.model",
    )
).expanduser()

#: 离散状态的分箱数（pi05：把 [-1, 1] 状态映射到 256 个整数）。
STATE_BINS = 256


class PaligemmaTokenizer:
    """把指令（以及 pi05 的离散状态）编码成定长 token 序列。"""

    def __init__(self, max_len: int = 200, model_path: str | pathlib.Path | None = None):
        path = pathlib.Path(model_path) if model_path is not None else DEFAULT_TOKENIZER_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"找不到 PaliGemma tokenizer：{path}。"
                "请设置环境变量 PALIGEMMA_TOKENIZER_PATH 指向 paligemma_tokenizer.model。"
            )
        with path.open("rb") as handle:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=handle.read())
        self._max_len = max_len

    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """返回 ``(tokens, mask)``，长度均为 ``max_len``。

        pi05 约定：状态先按 [-1, 1] 离散成 256 个整数，拼进语言前缀
        ``Task: ..., State: ...;\\nAction: ``；state=None 时用 pi0 的格式。
        """
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            discretized = np.digitize(state, bins=np.linspace(-1, 1, STATE_BINS + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")

        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if tokens_len > self._max_len:
                logger.warning(
                    "Token length (%d) exceeds max length (%d), truncating.",
                    tokens_len,
                    self._max_len,
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len
        return np.asarray(tokens, dtype=np.int64), np.asarray(mask, dtype=bool)


__all__ = ["DEFAULT_TOKENIZER_PATH", "STATE_BINS", "PaligemmaTokenizer"]
