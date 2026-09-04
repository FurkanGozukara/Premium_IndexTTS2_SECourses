"""Compatibility helpers for supported PyTorch dependency transitions."""

from __future__ import annotations

from enum import Enum
from functools import wraps
from threading import Lock
from typing import Any


_INSTALL_LOCK = Lock()
_PATCH_MARKER = "_indextts_native_enum_compatibility"


def install_native_enum_pytree_compatibility() -> bool:
    """Keep older dependencies from registering enums that PyTorch handles natively.

    PyTorch's opaque-object registry is the source of truth. On older PyTorch
    versions, or if that registry is unavailable, ``register_constant`` is left
    untouched so its original behavior is preserved.
    """

    try:
        from torch._library.opaque_object import is_opaque_type
        from torch.utils import _pytree
    except (ImportError, AttributeError):
        return False

    try:
        enums_are_native = bool(is_opaque_type(Enum))
    except (AttributeError, RuntimeError, TypeError):
        return False
    if not enums_are_native:
        return False

    with _INSTALL_LOCK:
        register_constant = getattr(_pytree, "register_constant", None)
        if not callable(register_constant):
            return False
        if getattr(register_constant, _PATCH_MARKER, False):
            return True

        @wraps(register_constant)
        def register_constant_compat(cls: type[Any]) -> None:
            if isinstance(cls, type) and issubclass(cls, Enum):
                try:
                    if is_opaque_type(cls):
                        return
                except (AttributeError, RuntimeError, TypeError):
                    pass
            register_constant(cls)

        setattr(register_constant_compat, _PATCH_MARKER, True)
        _pytree.register_constant = register_constant_compat
        return True


__all__ = ["install_native_enum_pytree_compatibility"]
