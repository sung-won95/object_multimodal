"""Compatibility helpers for moved top-level modules."""

from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType


def alias_module(compat_name: str, target_name: str) -> ModuleType:
    """Make an old module path resolve to the moved implementation module."""
    module = import_module(target_name)
    sys.modules[compat_name] = module

    parent_name, _, child_name = compat_name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None and child_name:
        setattr(parent, child_name, module)

    return module
