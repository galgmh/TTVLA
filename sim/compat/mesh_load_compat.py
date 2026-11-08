"""恢复 trimesh 5.0 之前对"包外"网格纹理的解析行为。

trimesh >= 5.0 默认 ``FilePathResolver(source, allow_anywhere=False)``：资源
不允许引用跳出自身目录的文件。robosuite 的纹理就放在网格目录之外，于是
加载时静默丢材质。
"""

from __future__ import annotations

import inspect


def install_texture_resolver_policy():
    """允许 OBJ 材质解析网格自身目录之外的纹理。"""
    from trimesh.resolvers import FilePathResolver

    original_init = FilePathResolver.__init__
    if getattr(original_init, "_pre5_allow_anywhere", False):
        return  # 已安装
    if "allow_anywhere" not in inspect.signature(original_init).parameters:
        return  # trimesh < 5：本来就没有目录限制

    def __init__(self, source, allow_anywhere=True):
        original_init(self, source, allow_anywhere=allow_anywhere)

    __init__._pre5_allow_anywhere = True
    FilePathResolver.__init__ = __init__
