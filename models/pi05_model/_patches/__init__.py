"""In-process injection of openpi-patched HuggingFace transformers modules.

The Pi0.5 PyTorch model needs modified versions of:
    transformers.models.siglip.modeling_siglip
    transformers.models.siglip.check        (sentinel module added by openpi)
    transformers.models.gemma.modeling_gemma
    transformers.models.gemma.configuration_gemma
    transformers.models.paligemma.modeling_paligemma

The official openpi recipe `cp`s these files into site-packages, polluting
the Python environment globally. We instead load them via
`importlib.util.spec_from_file_location` with `name="transformers.models.x.y"`,
which makes Python:
  * resolve relative imports inside each file (e.g. `from ...activations
    import ACT2FN`) against the real installed transformers, and
  * register the patched modules under the transformers namespace via
    `sys.modules`, so subsequent `from transformers import GemmaForCausalLM`
    resolves to the patched class.

Usage
-----
    import pi05_model
    pi05_model.install_patches()        # MUST run before importing PI0Pytorch
    from pi05_model import PI0Pytorch
    ...

The patches are version-locked to `transformers==4.53.2`. Pass
`strict_version=False` to downgrade the version mismatch into a warning.
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

_PATCH_MAP: dict[str, str] = {
    "transformers.models.siglip.modeling_siglip": "models/siglip/modeling_siglip.py",
    "transformers.models.siglip.check": "models/siglip/check.py",
    "transformers.models.gemma.modeling_gemma": "models/gemma/modeling_gemma.py",
    "transformers.models.gemma.configuration_gemma": "models/gemma/configuration_gemma.py",
    "transformers.models.paligemma.modeling_paligemma": "models/paligemma/modeling_paligemma.py",
}

_LAZY_TOP_LEVEL_NAMES = (
    "GemmaForCausalLM",
    "GemmaModel",
    "GemmaPreTrainedModel",
    "PaliGemmaForConditionalGeneration",
    "PaliGemmaModel",
    "PaliGemmaPreTrainedModel",
    "SiglipModel",
    "SiglipVisionModel",
)

_installed = False


def install(*, strict_version: bool = True) -> None:
    """Inject the patched modules into `sys.modules`. Idempotent.

    Must run BEFORE any code does `from transformers import GemmaForCausalLM`
    (or equivalent), otherwise the LazyModule inside `transformers` will have
    already cached the unpatched class objects.
    """
    global _installed
    if _installed:
        return

    import transformers

    if transformers.__version__ != _EXPECTED_TRANSFORMERS_VERSION:
        msg = (
            f"pi05_model._patches was built against transformers=="
            f"{_EXPECTED_TRANSFORMERS_VERSION}, but you have "
            f"{transformers.__version__}. Internal APIs used by the patches "
            f"may differ — the patched modules could fail to import."
        )
        if strict_version:
            raise RuntimeError(msg)
        warnings.warn(msg, stacklevel=2)

    # If the user previously did the file-cp recipe, the openpi sentinel
    # check.py will still sit inside site-packages. Harmless for us
    # (sys.modules wins), but worth flagging so they can clean up.
    tx_path = pathlib.Path(transformers.__file__).parent
    if (tx_path / "models" / "siglip" / "check.py").exists():
        warnings.warn(
            f"Detected leftover openpi sentinel at {tx_path/'models/siglip/check.py'}. "
            "Run `pip install --force-reinstall transformers==4.53.2` and "
            f"`rm {tx_path/'models/siglip/check.py'}` to restore a pristine env. "
            "The in-process patches in this package will continue to work either way.",
            stacklevel=2,
        )

    # Make sure parent packages exist as keys in sys.modules so we can hang
    # attributes off them below.
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
        sys.modules.pop(fqname, None)

        # `name=fqname` makes Python set `module.__package__` to
        # 'transformers.models.<x>', which is what makes relative imports
        # (e.g. `from ...activations import ACT2FN`) resolve against the
        # installed transformers package.
        spec = importlib.util.spec_from_file_location(fqname, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not build spec for {fqname} <- {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[fqname] = module
        spec.loader.exec_module(module)
        logger.debug("installed patched %s <- %s", fqname, path)

        parent_name, _, child_name = fqname.rpartition(".")
        setattr(sys.modules[parent_name], child_name, module)

    # Force transformers' LazyModule to re-resolve top-level names against
    # the patched modules. (No-op if these were never accessed yet.)
    tx_dict = sys.modules["transformers"].__dict__
    for name in _LAZY_TOP_LEVEL_NAMES:
        tx_dict.pop(name, None)

    _installed = True
    logger.info("pi05_model._patches: injected %d transformers modules", len(_PATCH_MAP))
