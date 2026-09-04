from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import subprocess
import sys

import pytest

from indextts.utils.torch_compat import install_native_enum_pytree_compatibility


ROOT = Path(__file__).resolve().parents[1]
DEPRECATION_TEXT = "register_constant() on Enum subclasses is deprecated"


def test_native_enum_compatibility_skips_only_enum_registration():
    pytest.importorskip("torch")
    try:
        from torch._library.opaque_object import is_opaque_type
        from torch.utils import _pytree
    except (ImportError, AttributeError):
        pytest.skip("PyTorch does not expose native opaque enum support")

    class NativeEnum(Enum):
        VALUE = "value"

    if not is_opaque_type(NativeEnum):
        pytest.skip("This PyTorch version does not handle enums natively")

    assert install_native_enum_pytree_compatibility() is True
    wrapped = _pytree.register_constant
    assert install_native_enum_pytree_compatibility() is True
    assert _pytree.register_constant is wrapped

    _pytree.register_constant(NativeEnum)
    assert not _pytree.is_constant_class(NativeEnum)
    assert _pytree.tree_flatten(NativeEnum.VALUE)[0] == [NativeEnum.VALUE]

    @dataclass(frozen=True)
    class RegularConstant:
        value: str

    try:
        _pytree.register_constant(RegularConstant)
        assert _pytree.is_constant_class(RegularConstant)
        leaves, spec = _pytree.tree_flatten(RegularConstant("kept"))
        assert leaves == []
        assert _pytree.tree_unflatten([], spec) == RegularConstant("kept")
    finally:
        if _pytree.is_constant_class(RegularConstant):
            _pytree._deregister_pytree_node(RegularConstant)


def test_ui_startup_import_has_no_obsolete_enum_registration_warning():
    environment = os.environ.copy()
    environment["PYTHONWARNINGS"] = "default"
    result = subprocess.run(
        [sys.executable, "-c", "import ui.app"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert DEPRECATION_TEXT not in result.stdout + result.stderr
