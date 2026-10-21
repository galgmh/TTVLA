"""
Restore trimesh's pre-5.0 resolution of out-of-tree mesh textures.

trimesh >= 5.0 defaults ``FilePathResolver(source, allow_anywhere=False)``: 
an asset may not reference a file that escapes its own directory.
"""

from __future__ import annotations

import inspect


def install_texture_resolver_policy():
    """
    Let OBJ materials resolve textures outside the mesh's own directory.
    """
    from trimesh.resolvers import FilePathResolver

    original_init = FilePathResolver.__init__
    if getattr(original_init, "_pre5_allow_anywhere", False):
        return  # already installed
    if "allow_anywhere" not in inspect.signature(original_init).parameters:
        return  # trimesh < 5: no root restriction to lift

    def __init__(self, source, allow_anywhere=True):
        original_init(self, source, allow_anywhere=allow_anywhere)

    __init__._pre5_allow_anywhere = True
    FilePathResolver.__init__ = __init__
