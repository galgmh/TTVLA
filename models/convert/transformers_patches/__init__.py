"""In-process replacement of HuggingFace transformers' Gemma/PaliGemma/SigLIP
modules with the openpi-patched versions, via `sys.modules` injection.

Rationale
---------
openpi's Pi0.5 PyTorch model needs modified versions of
  - transformers.models.siglip.modeling_siglip
  - transformers.models.siglip.check           (a sentinel file, openpi-only)
  - transformers.models.gemma.modeling_gemma
  - transformers.models.gemma.configuration_gemma
  - transformers.models.paligemma.modeling_paligemma

The upstream recipe is to physically `cp` the patched files over the installed
transformers package — which pollutes the Python environment globally.

Instead, we load the patched files from this directory using
`importlib.util.spec_from_file_location` with `name="transformers.models.xxx.yyy"`.
Setting that `name` makes Python resolve the relative imports inside each file
(e.g. `from ...activations import ACT2FN`) against the real installed
transformers package, while the patched classes end up in `sys.modules` under
the transformers namespace. Subsequent `from transformers import GemmaForCausalLM`
(triggered via transformers' LazyModule) then resolves to the patched class.

Usage
-----
    import transformers_patches
    transformers_patches.install()         # must run BEFORE importing PI0Pytorch
    # ... now safe to `import openpi.models_pytorch.pi0_pytorch` etc.

Version requirement
-------------------
The patches were produced against transformers==4.53.2 and use that version's
internal APIs (e.g. `GradientCheckpointingLayer`, ROPE helpers). `install()`
will raise if a different version is detected; pass `strict_version=False` to
downgrade that to a warning.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import pathlib
import sys
import warnings

logger = logging.getLogger(__name__)

_EXPECTED_TRANSFORMERS_VERSION = "4.53.2"

# (fully-qualified module name) -> path (relative to this package) of patched file.
_PATCH_MAP = {
    "transformers.models.siglip.modeling_siglip": "models/siglip/modeling_siglip.py",
    "transformers.models.siglip.check": "models/siglip/check.py",
    "transformers.models.gemma.modeling_gemma": "models/gemma/modeling_gemma.py",
    "transformers.models.gemma.configuration_gemma": "models/gemma/configuration_gemma.py",
    "transformers.models.paligemma.modeling_paligemma": "models/paligemma/modeling_paligemma.py",
}

_installed = False


def install(*, strict_version: bool = True) -> None:
    """Inject the patched modules into `sys.modules`.

    Idempotent: calling more than once is a no-op.

    Must be called before any code does
        `from transformers import GemmaForCausalLM`
    or equivalent, otherwise the LazyModule inside transformers will have
    already cached the unpatched class.
    """
    global _installed
    if _installed:
        return

    import transformers

    if transformers.__version__ != _EXPECTED_TRANSFORMERS_VERSION:
        msg = (
            f"transformers_patches was built for transformers=="
            f"{_EXPECTED_TRANSFORMERS_VERSION}, but you have "
            f"{transformers.__version__}. Internal APIs used by the patches "
            f"may differ — the patched modules could fail to import."
        )
        if strict_version:
            raise RuntimeError(msg)
        warnings.warn(msg, stacklevel=2)

    # Detect whether the user still has a file-level override in site-packages
    # (i.e. from the old `cp ./transformers_replace/* .../site-packages/transformers/`
    # recipe). If so, warn — it's harmless for us (sys.modules injection wins),
    # but cleaner to `pip install --force-reinstall transformers==4.53.2` to
    # restore a pristine env.
    tx_path = pathlib.Path(transformers.__file__).parent
    if (tx_path / "models" / "siglip" / "check.py").exists():
        warnings.warn(
            "Detected the openpi sentinel file "
            f"{tx_path / 'models' / 'siglip' / 'check.py'}; your site-packages "
            "transformers is still file-patched. You can reinstall it cleanly "
            "with `pip install --force-reinstall transformers==4.53.2` — the "
            "in-process patches in this package will continue to work either "
            "way.",
            stacklevel=2,
        )

    # Make sure the parent packages exist in sys.modules first, otherwise setting
    # an attribute on them below would fail.
    for parent in (
        "transformers.models",
        "transformers.models.gemma",
        "transformers.models.paligemma",
        "transformers.models.siglip",
    ):
        importlib.import_module(parent)

    patches_root = pathlib.Path(__file__).resolve().parent

    for fqname, relpath in _PATCH_MAP.items():
        path = patches_root / relpath
        if not path.exists():
            raise FileNotFoundError(f"Patched file missing: {path}")

        # Evict any cached original first so relative imports in the patched
        # file can't accidentally pick up the unpatched sibling.
        sys.modules.pop(fqname, None)

        # `name=fqname` sets the loaded module's __package__ to
        # 'transformers.models.xxx' — this is what makes relative imports like
        # `from ...activations import ACT2FN` resolve against the real
        # installed transformers.
        spec = importlib.util.spec_from_file_location(fqname, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not build spec for {fqname} <- {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[fqname] = module
        spec.loader.exec_module(module)
        logger.debug("installed patched %s <- %s", fqname, path)

        # Bind onto parent package as an attribute too (some code paths do
        # `transformers.models.gemma.modeling_gemma` as attribute access).
        parent_name, _, child_name = fqname.rpartition(".")
        setattr(sys.modules[parent_name], child_name, module)

    # Force transformers' LazyModule to re-resolve Gemma/PaliGemma symbols
    # against the patched modules. If `GemmaForCausalLM` was *already* grabbed
    # before install(), it'd be frozen to the unpatched class — there's no
    # clean way to retrofit that, hence the "call before importing PI0Pytorch"
    # contract.
    for name in ("GemmaForCausalLM", "GemmaModel", "GemmaPreTrainedModel"):
        sys.modules["transformers"].__dict__.pop(name, None)
    for name in (
        "PaliGemmaForConditionalGeneration",
        "PaliGemmaModel",
        "PaliGemmaPreTrainedModel",
    ):
        sys.modules["transformers"].__dict__.pop(name, None)
    for name in ("SiglipModel", "SiglipVisionModel"):
        sys.modules["transformers"].__dict__.pop(name, None)

    _installed = True
    logger.info("transformers_patches: injected %d modules", len(_PATCH_MAP))
