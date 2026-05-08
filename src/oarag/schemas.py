"""Compatibility wrapper for :mod:`oarag.core.schemas`."""

from oarag._compat import alias_module as _alias_module

_module = _alias_module(__name__, "oarag.core.schemas")
globals().update(_module.__dict__)
